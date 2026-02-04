import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict, List, Tuple, Any, Optional

# =========================
# 1. 核心數學工具 (V34.0 Supercharged)
# =========================
EPS = 1e-15

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def nb_pmf(k: int, mu: float, alpha: float) -> float:
    if alpha <= 0: return poisson_pmf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    coeff = math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1))
    return float(coeff * (p ** r) * ((1 - p) ** k))

def biv_poisson_pmf(i: int, j: int, lam1: float, lam2: float, lam3: float) -> float:
    """[V34] 雙變量 Poisson P(X=i, Y=j) with shared lambda3 (Covariance)"""
    if (lam1 + lam2 + lam3) <= 0: return 1.0 if (i == 0 and j == 0) else 0.0
    terms = []
    base = -(lam1 + lam2 + lam3)
    maxk = min(i, j)
    for k in range(0, maxk + 1):
        lt = base
        if (i - k) > 0: lt += (i - k) * math.log(lam1) - math.lgamma(i - k + 1)
        if (j - k) > 0: lt += (j - k) * math.log(lam2) - math.lgamma(j - k + 1)
        if k > 0: lt += k * math.log(lam3) - math.lgamma(k + 1)
        terms.append(lt)
    m = max(terms)
    s = sum(math.exp(t - m) for t in terms)
    return max(math.exp(m + math.log(s)), 0.0)

def get_true_implied_prob(odds_dict: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / float(v) if v > 0 else 0.0 for k, v in odds_dict.items()}
    margin = sum(inv.values())
    return {k: inv[k] / margin if margin > 0 else 0.0 for k in odds_dict}

def calc_risk_adj_kelly(ev_percent: float, variance: float, risk_scale: float = 0.5, prob: float = 0.5) -> float:
    if variance <= 0 or ev_percent <= 0: return 0.0
    ev = ev_percent / 100.0
    f = (ev / variance) * risk_scale
    cap = 0.5 if prob >= 0.35 else 0.025
    return min(cap, max(0.0, f)) * 100

def calc_risk_metrics(prob: float, odds: float) -> Tuple[float, float]:
    if prob <= 0 or prob >= 1: return 0.0, 0.0
    win_p, lose_p = odds - 1.0, -1.0
    ev = prob * win_p + (1 - prob) * lose_p
    var = prob * (win_p**2) + (1 - prob) * (lose_p**2) - (ev**2)
    sharpe = ev / math.sqrt(var) if var > 0 else 0
    return var, sharpe

# =========================
# 2. 校正與分析工具 (V34 New Labs)
# =========================

def build_biv_matrix(lh: float, la: float, lam3: float, max_g: int) -> np.ndarray:
    """[V34] 構建雙變量矩陣"""
    G = max_g
    M = np.zeros((G, G), dtype=float)
    # 若 lam3 為 0，退化為獨立 Poisson
    if lam3 <= 1e-9:
        i, j = np.arange(G), np.arange(G)
        p_i = np.array([poisson_pmf(k, lh) for k in i])
        p_j = np.array([poisson_pmf(k, la) for k in j])
        M = np.outer(p_i, p_j)
    else:
        # 使用 biv_poisson_pmf (需注意這比 numpy vectorization 慢，但在 Streamlit 可接受)
        for i in range(G):
            for j in range(G):
                M[i, j] = biv_poisson_pmf(i, j, lh, la, lam3)
    s = M.sum()
    return M / max(s, EPS)

def neg_loglike_grid(history_df: pd.DataFrame, lam3: float, nb_alpha: float, w: float, max_g: int = 9) -> float:
    """計算負對數概似 (Calibration Objective)"""
    nll = 0.0
    for _, row in history_df.iterrows():
        lh, la = float(row['lh_pred']), float(row['la_pred'])
        i, j = int(row['home_goals']), int(row['away_goals'])
        
        # 這裡簡化：混合模型 = w * Biv(lam3) + (1-w) * Indep(Poisson)
        # 為了速度，這裡假設 Calibration 時 market implied 也是一種 Poisson
        M_biv = build_biv_matrix(lh, la, lam3, max_g)
        
        # 處理邊界
        prob = 0.0
        if i < max_g and j < max_g: prob = M_biv[i, j]
        else: prob = 1e-6 # 簡單處理溢出
        
        # 混合參數 w (若 w < 1 代表考慮了雜訊/市場)
        # 這裡簡單模擬：混合一個較為平坦的分布 (Uniform or Indep NB)
        prob = w * prob + (1-w) * (1.0/(max_g*max_g)) 
        
        nll -= math.log(max(prob, EPS))
    return nll

def calibrate_grid_search(history_df: pd.DataFrame) -> Dict[str, Any]:
    """[V34] 參數網格搜索"""
    # 簡化網格以加快演示速度
    lam3_rng = [0.0, 0.1, 0.3]
    w_rng = [0.6, 0.8, 1.0]
    best = {"nll": float("inf"), "lam3": 0.0, "w": 0.8}
    
    for lam3 in lam3_rng:
        for w in w_rng:
            nll = neg_loglike_grid(history_df, lam3, 0.12, w)
            if nll < best["nll"]:
                best = {"nll": nll, "lam3": lam3, "w": w}
    return best

def importance_sampling_over(M: np.ndarray, line: float, n_sims: int = 10000, bias_power: float = 1.5) -> Dict[str, Any]:
    """[V34] 重要性採樣 (針對高賠率 OU)"""
    rng = np.random.default_rng(42)
    G = M.shape[0]
    flat = M.flatten()
    idx = np.arange(G*G); i = idx // G; j = idx % G
    sums = (i + j).astype(float)
    
    # Bias: 讓高比分更容易被選中
    bias = (1.0 + sums) ** bias_power
    q = flat * bias; q /= q.sum()
    
    draws = rng.choice(G*G, size=n_sims, p=q)
    chosen_sums = sums[draws]
    weights = flat[draws] / q[draws] # Importance weights
    
    indicators = (chosen_sums > line).astype(float)
    est = np.sum(weights * indicators) / np.sum(weights)
    ess = 1.0 / np.sum((weights / weights.sum()) ** 2)
    return {"est": float(est), "ess": float(ess)}

# =========================
# 3. 全景記憶體系 (V32 保留)
# =========================
class RegimeMemory:
    def __init__(self):
        self.history_db = {
            "Bore_Draw_Stalemate": { "name": "🛡️ 雙重鐵桶", "bets": 19, "roi": 0.219 }, 
            "Relegation_Dog": { "name": "🐕 保級受讓", "bets": 101, "roi": 0.083 },
            "Fallen_Giant": { "name": "📉 豪門崩盤", "bets": 67, "roi": -0.008 },
            "Fortress_Home": { "name": "🏰 魔鬼主場", "bets": 256, "roi": -0.008 },
            "Title_MustWin_Home": { "name": "🏆 爭冠必勝盤", "bets": 256, "roi": -0.063 },
            "MarketHype_Fav": { "name": "🔥 大熱倒灶", "bets": 150, "roi": -0.080 },
            "MidTable_Standard": { "name": "😐 中游例行", "bets": 300, "roi": 0.000 }
        }

    def analyze_scenario(self, lh, la, odds) -> str:
        if odds["home"] < 1.30: return "MarketHype_Fav"
        if (lh + la) < 2.2: return "Bore_Draw_Stalemate"
        return "MidTable_Standard"

    def recall_experience(self, regime_id: str) -> Dict:
        return self.history_db.get(regime_id, {"name": "未知", "roi": 0.0, "bets":0})

    def calc_memory_penalty(self, historical_roi: float) -> float:
        if historical_roi < -0.05: return 0.7
        if historical_roi > 0.05: return 1.1
        return 1.0

# =========================
# 4. 分析引擎邏輯 (V34.0 Supercharged Core)
# =========================
class SniperAnalystLogic:
    def __init__(self, json_data: Any, max_g: int = 9, nb_alpha: float = 0.12, lam3: float = 0.0):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha
        self.lam3 = lam3 # [V34] 共變異數參數
        self.memory = RegimeMemory()

    def calc_lambda(self) -> Tuple[float, float, bool]:
        """計算 Lambda (含近況加權)"""
        league_base = 1.35
        is_weighted = False
        def att_def_w(team):
            nonlocal is_weighted
            xg, xga = team["offensive_stats"].get("xg_avg", team["offensive_stats"]["goals_scored_avg"]), team["defensive_stats"].get("xga_avg", team["defensive_stats"]["goals_conceded_avg"])
            trend = team["context_modifiers"].get("recent_form_trend", [0, 0, 0])
            if any(t != 0 for t in trend): is_weighted = True
            w = np.array([0.1, 0.3, 0.6])
            form_factor = 1.0 + (np.dot(trend[-len(w):], w[-len(trend):]) * 0.1)
            return (0.3 * team["offensive_stats"]["goals_scored_avg"] + 0.7 * xg) * form_factor, (0.3 * team["defensive_stats"]["goals_conceded_avg"] + 0.7 * xga)

        lh_att, lh_def = att_def_w(self.h)
        la_att, la_def = att_def_w(self.a)
        if self.h["context_modifiers"].get("missing_key_defender"): lh_def *= 1.25
        if self.a["context_modifiers"].get("missing_key_defender"): la_def *= 1.20
        h_adv = self.h["general_strength"].get("home_advantage_weight", 1.15)
        return (lh_att * la_def / league_base) * h_adv, (la_att * lh_def / league_base), is_weighted

    def build_matrix_v34(self, lh: float, la: float, use_bivariate: bool = False) -> Tuple[np.ndarray, Dict]:
        """[V34] 矩陣生成 (支援 Bivariate)"""
        if use_bivariate and self.lam3 > 0:
            # 1. 使用 Bivariate Poisson 矩陣
            M_model = build_biv_matrix(lh, la, self.lam3, self.max_g)
        else:
            # 2. V33 原版混合矩陣 (Indep Poisson + NB)
            M_model = get_matrix_cached(lh, la, self.max_g, self.nb_alpha, False) # 簡化調用

        true_imp = get_true_implied_prob(self.market["1x2_odds"])
        p_h, p_d, p_a = float(np.sum(np.tril(M_model, -1))), float(np.sum(np.diag(M_model))), float(np.sum(np.triu(M_model, 1)))
        
        # 混合疊加權重: Model 70% + Market 30%
        w = 0.7
        t_h, t_d, t_a = w*p_h + (1-w)*true_imp["home"], w*p_d + (1-w)*true_imp["draw"], w*p_a + (1-w)*true_imp["away"]
        
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
        # 簡易敏感度測試
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
    
    # [V34] 參數擾動以產生不確定性區間
    def simulate_uncertainty(self, lh, la, base_ev):
        # 簡單模擬：假設 lambda 有 +/- 10% 的誤差
        evs = []
        for _ in range(50):
            lh_s = lh * np.random.normal(1.0, 0.1)
            la_s = la * np.random.normal(1.0, 0.1)
            # 快速估計不做完整矩陣重建，僅做比例縮放
            ratio = (lh_s - la_s) / (lh - la) if abs(lh - la) > 0.1 else 1.0
            evs.append(base_ev * ratio)
        return np.percentile(evs, 5), np.percentile(evs, 95)

# =========================
# 5. Streamlit UI (V34 Ultimate)
# =========================
st.set_page_config(page_title="Sniper V34.0 Supercharged", page_icon="🚀", layout="wide")

# CSS
st.markdown("""
<style>
    .metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.title("🚀 Sniper V34.0")
    st.caption("Supercharged Edition")
    st.markdown("---")
    app_mode = st.radio("功能模式：", ["🎯 單場深度預測", "🔧 參數校正實驗室", "📈 聯賽歷史回測", "📚 劇本查詢"])
    
    st.divider()
    with st.expander("🛠️ 進階參數 (V34)", expanded=False):
        unit_stake = st.number_input("單注本金 ($)", 10, 10000, 100)
        nb_alpha = st.slider("Alpha (NB)", 0.05, 0.25, 0.12)
        
        # [V34] Bivariate Control
        use_biv = st.toggle("啟用 Bivariate Poisson", value=False)
        lam3_input = st.slider("共變異數 (Lambda 3)", 0.0, 0.5, 0.1, help="當雙方進球相關性高時(如天氣差)調高此值") if use_biv else 0.0
        
        risk_scale = st.slider("風險係數", 0.1, 1.0, 0.3)
        use_mock_memory = st.checkbox("歷史記憶修正", value=True)
        show_uncertainty = st.toggle("顯示 EV 不確定區間", value=True)

# =========================
# 模式 1: 單場深度預測
# =========================
if app_mode == "🎯 單場深度預測":
    st.header("🎯 單場深度預測 (V34 Engine)")
    
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = None

    tab1, tab2 = st.tabs(["📋 貼上 JSON", "📂 上傳 JSON"])
    input_data = None
    with tab1:
        j_txt = st.text_area("在此貼上 JSON", height=100)
        if j_txt: 
            try: input_data = json.loads(j_txt)
            except: st.error("JSON 格式錯誤")
    with tab2:
        u_file = st.file_uploader("選擇檔案", type=['json'])
        if u_file: input_data = json.load(u_file)

    if st.button("🚀 執行分析", type="primary"):
        if input_data:
            # 初始化 V34 引擎 (傳入 lam3)
            engine = SniperAnalystLogic(input_data, 9, nb_alpha, lam3_input)
            
            lh, la, is_weighted = engine.calc_lambda()
            # 構建矩陣 (支援 Bivariate)
            M, probs_detail = engine.build_matrix_v34(lh, la, use_bivariate=use_biv)
            
            market_bonus = engine.get_market_trend_bonus()
            regime_id = engine.memory.analyze_scenario(engine, lh, la)
            history_data = engine.memory.recall_experience(regime_id)
            penalty = engine.memory.calc_memory_penalty(history_data["roi"]) if use_mock_memory else 1.0
            
            p_h = probs_detail["hybrid"]["home"]
            m_h = probs_detail["market"]["home"]
            diff_p = abs(p_h - m_h) / max(m_h, 1e-9)
            sens_lv, sens_dr = engine.check_sensitivity(lh, la)
            conf_score, conf_reasons = engine.calc_model_confidence(lh, la, diff_p, sens_dr)

            st.session_state.analysis_results = {
                "engine": engine, "M": M, "lh": lh, "la": la, "is_weighted": is_weighted,
                "probs_detail": probs_detail, "market_bonus": market_bonus,
                "history_data": history_data, "penalty": penalty,
                "conf_score": conf_score, "conf_reasons": conf_reasons
            }
        else:
            st.error("請輸入數據")

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        engine, M = res["engine"], res["M"]
        probs = res["probs_detail"]

        # --- 儀表板 ---
        st.markdown("### 🔍 V34 戰術儀表板")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("主隊進球預期", f"{res['lh']:.2f}", delta="加權啟用" if res["is_weighted"] else None)
        d2.metric("客隊進球預期", f"{res['la']:.2f}")
        
        hybrid_p = probs["hybrid"]["home"]
        model_p = probs["model"]["home"]
        d3.metric("V34 混合主勝", f"{hybrid_p:.1%}", delta=f"{(hybrid_p - model_p)*100:+.1f}%", delta_color="inverse")
        
        conf_score = res["conf_score"]
        d4.metric("🛡️ 信心指數", f"{conf_score:.0%}")
        
        if conf_score < 1.0:
            with st.expander(f"⚠️ 信心扣分診斷", expanded=True):
                for r in res["conf_reasons"]: st.warning(f"🔻 {r}")

        # --- 分頁分析 ---
        t_val, t_ai, t_score, t_sim = st.tabs(["💰 價值投資", "🧠 智能裁決", "🎯 波膽分佈", "🎲 進階模擬"])
        
        candidates = [] 
        
        with t_val:
            st.subheader("獨贏 (1x2)")
            r_1x2 = []
            for tag, key in [("主勝", "home"), ("和局", "draw"), ("客勝", "away")]:
                prob = probs["hybrid"][key]
                odd = engine.market["1x2_odds"][key]
                raw_ev = (prob * odd - 1) * 100 + res["market_bonus"][key]
                adj_ev = raw_ev * conf_score * res["penalty"]
                
                # [V34] 不確定性區間
                ev_low, ev_high = engine.simulate_uncertainty(res['lh'], res['la'], adj_ev) if show_uncertainty else (adj_ev, adj_ev)
                ev_str = f"{adj_ev:+.1f}%"
                if show_uncertainty: ev_str += f" [{ev_low:.1f}, {ev_high:.1f}]"

                var, sharpe = calc_risk_metrics(prob, odd)
                kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, prob)
                
                r_1x2.append({"選項": tag, "賠率": odd, "修正EV": ev_str, "注碼%": f"{kelly:.1f}%"})
                if adj_ev > 1.0: candidates.append({"pick": tag, "odds": odd, "ev": adj_ev, "kelly": kelly, "type": "1x2", "sharpe": sharpe})
            st.dataframe(pd.DataFrame(r_1x2), use_container_width=True)
            
            c_ah, c_ou = st.columns(2)
            with c_ah:
                st.subheader("亞盤 (AH)")
                target_o = engine.market.get("target_odds", 1.90)
                rows_ah = []
                for hcap in engine.market["handicaps"]:
                    raw_ev = engine.ah_ev(M, hcap, target_o) + res["market_bonus"]["home"]
                    adj_ev = raw_ev * conf_score * res["penalty"]
                    prob_apx = (raw_ev/100.0 + 1) / target_o
                    var, sharpe = calc_risk_metrics(prob_apx, target_o)
                    kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, prob_apx)
                    rows_ah.append({"盤口": f"主 {hcap:+}", "賠率": target_o, "修正EV": f"{adj_ev:+.1f}%", "注碼%": f"{kelly:.1f}%"})
                    if adj_ev > 1.5: candidates.append({"pick": f"主 {hcap:+}", "odds": target_o, "ev": adj_ev, "kelly": kelly, "type": "AH", "sharpe": sharpe})
                st.dataframe(pd.DataFrame(rows_ah), use_container_width=True)

            with c_ou:
                st.subheader("大小 (OU)")
                rows_ou = []
                idx_sum = np.add.outer(np.arange(engine.max_g), np.arange(engine.max_g))
                for line in engine.market["goal_lines"]:
                    p_over = float(M[idx_sum > line].sum())
                    raw_ev = (p_over * target_o - 1) * 100
                    adj_ev = raw_ev * conf_score * res["penalty"]
                    var, sharpe = calc_risk_metrics(p_over, target_o)
                    kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, p_over)
                    rows_ou.append({"盤口": f"大 {line}", "賠率": target_o, "修正EV": f"{adj_ev:+.1f}%", "注碼%": f"{kelly:.1f}%"})
                    if adj_ev > 1.5: candidates.append({"pick": f"大 {line}", "odds": target_o, "ev": adj_ev, "kelly": kelly, "type": "OU", "sharpe": sharpe})
                st.dataframe(pd.DataFrame(rows_ou), use_container_width=True)

            st.divider()
            st.markdown("### 🏆 智能投資組合")
            if candidates:
                best_picks = sorted(candidates, key=lambda x: x['ev'], reverse=True)[:3]
                reco_data = []
                for p in best_picks:
                    amt = unit_stake * (p['kelly'] / 100.0)
                    icon = "🟢" if p['sharpe'] > 0.1 else "🟡"
                    reco_data.append([f"[{p['type']}] {p['pick']}", p['odds'], f"{p['ev']:+.1f}%", f"{icon} {p['sharpe']:.2f}", f"{p['kelly']:.1f}%", f"${amt:.1f}"])
                st.dataframe(pd.DataFrame(reco_data, columns=["選項", "賠率", "EV", "夏普", "注碼", "金額"]), use_container_width=True)
                top = best_picks[0]
                st.success(f"🔥 首選推薦：**{top['pick']}**")
            else:
                st.info("🚧 本場比賽風險過高或 EV 不足，建議觀望。")

        with t_ai:
            st.write("V34 權重混合分析")
            df_comp = pd.DataFrame([probs["model"], probs["market"], probs["hybrid"]], index=["純模型", "市場去水", "V34混合"])
            st.dataframe(df_comp.style.format("{:.1%}"))

        with t_score:
            st.write("波膽矩陣 (Hybrid)")
            st.dataframe(pd.DataFrame(M[:6,:6]).style.format("{:.1%}"))

        with t_sim:
            # [V34] 顯示 Importance Sampling 與一般 MC
            st.subheader("稀有事件估計 (Importance Sampling)")
            line_check = 4.5
            is_res = importance_sampling_over(M, line_check)
            st.metric(f"大 {line_check} 機率 (IS估計)", f"{is_res['est']:.2%}", help=f"ESS: {is_res['ess']:.0f}")
            
            st.divider()
            st.subheader("一般蒙地卡羅 (10,000 runs)")
            sh, sa, sr = engine.run_monte_carlo(M)
            st.metric("主勝率 (MC)", f"{sr.count('home')/100:.1f}%")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(sh, alpha=0.5, label="Home"); ax.hist(sa, alpha=0.5, label="Away"); ax.legend()
            st.pyplot(fig)

# =========================
# 模式 2: 參數校正實驗室 (V34 新功能)
# =========================
elif app_mode == "🔧 參數校正實驗室":
    st.header("🔧 參數校正實驗室 (Calibration Lab)")
    st.markdown("上傳歷史回測 CSV，利用 Grid Search 尋找最佳 Lambda3 (共變異) 與 Alpha")
    
    cal_file = st.file_uploader("上傳含有 lh_pred, la_pred, home_goals, away_goals 的 CSV", type=['csv'])
    
    if cal_file:
        df_cal = pd.read_csv(cal_file)
        st.write("預覽數據:", df_cal.head())
        
        if st.button("⚡ 開始 Grid Search", type="primary"):
            with st.spinner("正在進行參數網格搜索..."):
                # 呼叫 V34 校正函數
                best_params = calibrate_grid_search(df_cal)
            
            st.success("校正完成！最佳參數如下：")
            c1, c2, c3 = st.columns(3)
            c1.metric("最佳 Lambda3", f"{best_params['lam3']:.2f}")
            c2.metric("最佳混合權重 w", f"{best_params['w']:.2f}")
            c3.metric("最小 NLL", f"{best_params['nll']:.4f}")
            st.info("💡 建議將這些參數更新到側邊欄設定中。")
            
    else:
        # 產生模擬數據供測試
        if st.button("生成模擬測試數據"):
            mock_df = pd.DataFrame({
                'lh_pred': np.random.uniform(1.0, 2.5, 50),
                'la_pred': np.random.uniform(0.8, 2.0, 50),
                'home_goals': np.random.randint(0, 5, 50),
                'away_goals': np.random.randint(0, 4, 50)
            })
            st.dataframe(mock_df)
            st.caption("請將此數據存為 CSV 後上傳測試。")

# =========================
# 模式 3 & 4 (保留)
# =========================
elif app_mode == "📈 聯賽歷史回測":
    st.title("📈 聯賽歷史回測")
    st.info("請將 CSV 檔案放入資料夾後，使用 V34 Batch Engine 進行測試。")

elif app_mode == "📚 劇本查詢":
    st.title("📚 盤口劇本庫")
    mem = RegimeMemory()
    data = [{"劇本": v["name"], "ROI": f"{v['roi']:.1%}", "樣本": v["bets"]} for k, v in mem.history_db.items()]
    st.dataframe(pd.DataFrame(data), use_container_width=True)
