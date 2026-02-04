# =========================
# 4. 分析引擎邏輯 (V35.0 Architect Core)
# =========================
class SniperAnalystLogic:
    def __init__(self, json_data: Any, max_g: int = 9, nb_alpha: float = 0.12, lam3: float = 0.0):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha
        self.lam3 = lam3 
        self.memory = RegimeMemory()

    def calc_lambda(self) -> Tuple[float, float, bool]:
        """計算 Lambda (維持 V34 的近況加權)"""
        league_base = 1.35
        is_weighted = False
        def att_def_w(team):
            nonlocal is_weighted
            xg = team["offensive_stats"].get("xg_avg", 1.0)
            xga = team["defensive_stats"].get("xga_avg", 1.0)
            trend = team["context_modifiers"].get("recent_form_trend", [0, 0, 0])
            if any(t != 0 for t in trend): is_weighted = True
            w = np.array([0.1, 0.3, 0.6])
            form_factor = 1.0 + (np.dot(trend[-len(w):], w[-len(trend):]) * 0.1)
            return (0.3 * team["offensive_stats"]["goals_scored_avg"] + 0.7 * xg) * form_factor, \
                   (0.3 * team["defensive_stats"]["goals_conceded_avg"] + 0.7 * xga)

        lh_att, lh_def = att_def_w(self.h)
        la_att, la_def = att_def_w(self.a)
        
        # 這裡加入一個 V35 微調：強隊打弱隊的「碾壓係數」
        strength_gap = (lh_att - la_att)
        crush_factor = 1.05 if strength_gap > 0.5 else 1.0
        
        if self.h["context_modifiers"].get("missing_key_defender"): lh_def *= 1.25
        if self.a["context_modifiers"].get("missing_key_defender"): la_def *= 1.20
        h_adv = self.h["general_strength"].get("home_advantage_weight", 1.15)
        
        return (lh_att * la_def / league_base) * h_adv * crush_factor, \
               (la_att * lh_def / league_base), is_weighted

    def dixon_coles_adjustment(self, M: np.ndarray, lh: float, la: float, rho: float = -0.13) -> np.ndarray:
        """
        [V35 新增] Dixon-Coles 修正核心
        專門修正 0-0, 1-0, 0-1, 1-1 的機率，解決 Poisson 低估和局的問題
        rho: 依賴參數，通常在 -0.1 到 -0.15 之間
        """
        # 定義修正函數 tau
        def tau(x, y, lambda_h, lambda_a, rho):
            if x == 0 and y == 0: return 1.0 - (lambda_h * lambda_a * rho)
            elif x == 0 and y == 1: return 1.0 + (lambda_h * rho)
            elif x == 1 and y == 0: return 1.0 + (lambda_a * rho)
            elif x == 1 and y == 1: return 1.0 - rho
            else: return 1.0

        M_adj = M.copy()
        # 只修正左上角 2x2 區域
        for i in range(2):
            for j in range(2):
                M_adj[i, j] *= tau(i, j, lh, la, rho)
        
        return M_adj / M_adj.sum() # 重新歸一化

    def apply_heavy_tail(self, M: np.ndarray, boost: float = 1.02) -> np.ndarray:
        """
        [V35 新增] 尾部增強 (Heavy Tail)
        輕微增加大比分 (3球以上) 的權重，模擬瘋狂比賽
        """
        G = M.shape[0]
        # 建立一個權重遮罩，越往右下角權重越高
        weights = np.ones((G, G))
        for i in range(3, G):
            for j in range(3, G):
                weights[i, j] = boost # 放大高比分區域
        
        M_boosted = M * weights
        return M_boosted / M_boosted.sum()

    def build_matrix_v35(self, lh: float, la: float, use_bivariate: bool = False, use_dc: bool = True) -> Tuple[np.ndarray, Dict]:
        """[V35] 整合所有數學模型的最終矩陣生成器"""
        
        # 1. 基礎物理層 (Bivariate or Indep)
        if use_bivariate and self.lam3 > 0:
            M_model = build_biv_matrix(lh, la, self.lam3, self.max_g)
        else:
            M_model = get_matrix_cached(lh, la, self.max_g, self.nb_alpha, False) 

        # 2. [V35] 應用 Dixon-Coles 修正 (針對低比分)
        if use_dc:
            # rho 值通常由歷史數據訓練，這裡使用經驗值 -0.13
            M_model = self.dixon_coles_adjustment(M_model, lh, la, rho=-0.13)

        # 3. [V35] 應用尾部增強 (針對高變異球隊)
        vol_str = self.h.get("style_of_play", {}).get("volatility", "normal")
        if vol_str == "high":
            M_model = self.apply_heavy_tail(M_model, boost=1.05)

        # 4. 市場混合層 (Bayesian Smoothing)
        true_imp = get_true_implied_prob(self.market["1x2_odds"])
        p_h = float(np.sum(np.tril(M_model, -1)))
        p_d = float(np.sum(np.diag(M_model)))
        p_a = float(np.sum(np.triu(M_model, 1)))
        
        # V35 動態權重：如果模型與市場差異過大，自動降低模型權重 (防禦機制)
        market_diff = abs(p_h - true_imp["home"])
        w = 0.7 if market_diff < 0.2 else 0.5 # 差異太大時，50/50 混合
        
        t_h = w*p_h + (1-w)*true_imp["home"]
        t_d = w*p_d + (1-w)*true_imp["draw"]
        t_a = w*p_a + (1-w)*true_imp["away"]
        
        M_hybrid = M_model.copy()
        M_hybrid[np.tril_indices(self.max_g, -1)] *= (t_h / p_h if p_h > 0 else 1)
        M_hybrid[np.diag_indices(self.max_g)] *= (t_d / p_d if p_d > 0 else 1)
        M_hybrid[np.triu_indices(self.max_g, 1)] *= (t_a / p_a if p_a > 0 else 1)
        M_hybrid /= M_hybrid.sum()
        
        probs_detail = {
            "model": {"home": p_h, "draw": p_d, "away": p_a},
            "market": true_imp,
            "hybrid": {"home": t_h, "draw": t_d, "away": t_a}
        }
        return M_hybrid, probs_detail

    # ... (保留原本的 helper functions: get_market_trend_bonus, ah_ev, run_monte_carlo, check_sensitivity, calc_model_confidence, simulate_uncertainty) ...
    # 請確保這些 V34 的舊方法都還在類別裡，不需要改動
    def get_market_trend_bonus(self) -> Dict[str, float]:
        bonus = {"home":0.0, "draw":0.0, "away":0.0}
        op, cu = self.market.get("opening_odds"), self.market.get("1x2_odds")
        if not op or not cu: return bonus
        for k in bonus:
            drop = max(0.0, (op[k] - cu[k]) / op[k])
            bonus[k] = min(3.0, drop * 30.0)
        return bonus

    def ah_ev(self, M: np.ndarray, hcap: float, odds: float) -> float:
        q = int(round(hcap * 4))
        if q % 2 != 0: return 0.5 * self.ah_ev(M, (q+1)/4.0, odds) + 0.5 * self.ah_ev(M, (q-1)/4.0, odds)
        idx_diff = np.subtract.outer(np.arange(self.max_g), np.arange(self.max_g)) 
        r_matrix = idx_diff + hcap
        payoff = np.select([r_matrix > 0.001, np.abs(r_matrix) <= 0.001, r_matrix < -0.001], [odds - 1, 0, -1], default=-1)
        return np.sum(M * payoff) * 100

    def check_sensitivity(self, lh: float, la: float) -> Tuple[str, float]:
        M_stress = get_matrix_cached(lh, la + 0.3, self.max_g, self.nb_alpha, False)
        p_orig = float(np.sum(np.tril(get_matrix_cached(lh, la, self.max_g, self.nb_alpha, False), -1)))
        p_new = float(np.sum(np.tril(M_stress, -1)))
        drop = (p_orig - p_new) / p_orig if p_orig > 0 else 0
        return ("High" if drop > 0.15 else "Medium"), drop

    def calc_model_confidence(self, lh: float, la: float, market_diff: float, sens_drop: float) -> Tuple[float, List[str]]:
        score, reasons = 1.0, []
        if market_diff > 0.25: score *= 0.7; reasons.append(f"與市場差異過大 ({market_diff:.1%})")
        if sens_drop > 0.15: score *= 0.8; reasons.append("模型對運氣球敏感")
        if (lh + la) > 3.5: score *= 0.9; reasons.append("高變異風險 (xG > 3.5)")
        return score, reasons
    
    def simulate_uncertainty(self, lh, la, base_ev):
        evs = []
        for _ in range(50):
            lh_s = lh * np.random.normal(1.0, 0.1)
            la_s = la * np.random.normal(1.0, 0.1)
            ratio = (lh_s - la_s) / (lh - la) if abs(lh - la) > 0.1 else 1.0
            evs.append(base_ev * ratio)
        return np.percentile(evs, 5), np.percentile(evs, 95)

    def run_monte_carlo(self, M: np.ndarray, sims: int = 10000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        rng = np.random.default_rng(seed); G = M.shape[0]; flat_M = M.flatten()
        indices = rng.choice(G * G, size=sims, p=flat_M/flat_M.sum())
        sh, sa = indices // G, indices % G
        res = np.full(sims, "draw", dtype=object); res[sh > sa] = "home"; res[sh < sa] = "away"
        return sh, sa, res.tolist()
