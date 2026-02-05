import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize

# =========================
# 1. 核心數學工具 (V37.0 Omni-Kernel)
# =========================
EPS = 1e-15

@lru_cache(maxsize=2048)
def log_factorial(n: int) -> float:
    """使用 gammaln 進行快速階乘對數計算"""
    return gammaln(n + 1)

def poisson_logpmf(k: int, lam: float) -> float:
    """對數空間 Poisson PMF"""
    if lam <= 0: return 0.0 if k == 0 else -np.inf
    return -lam + k * math.log(lam) - log_factorial(k)

def biv_poisson_logpmf(x: int, y: int, lam1: float, lam2: float, lam3: float) -> float:
    """[V37] 對數空間雙變量 Poisson (數值絕對穩定)"""
    if lam3 <= 1e-9:
        return poisson_logpmf(x, lam1) + poisson_logpmf(y, lam2)
    
    base = -(lam1 + lam2 + lam3)
    terms = []
    min_k = min(x, y)
    
    for k in range(min_k + 1):
        try:
            t = base
            if x - k > 0: t += (x - k) * math.log(lam1) - log_factorial(x - k)
            if y - k > 0: t += (y - k) * math.log(lam2) - log_factorial(y - k)
            if k > 0: t += k * math.log(lam3) - log_factorial(k)
            terms.append(t)
        except ValueError: continue
            
    return logsumexp(terms)

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

@st.cache_data
def get_matrix_cached(lh: float, la: float, max_g: int, nb_alpha: float) -> np.ndarray:
    """Legacy Matrix Builder (Fallback)"""
    G = max_g
    M = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            p = math.exp(poisson_logpmf(i, lh) + poisson_logpmf(j, la))
            M[i, j] = p
    return M / M.sum()

# =========================
# 2. 全景記憶體系 (Fixed)
# =========================
class RegimeMemory:
    def __init__(self):
        self.history_db = {
            "Bore_Draw_Stalemate": { "name": "🛡️ 雙重鐵桶", "roi": 0.219 }, 
            "Relegation_Dog": { "name": "🐕 保級受讓", "roi": 0.083 },
            "Fallen_Giant": { "name": "📉 豪門崩盤", "roi": -0.008 },
            "Fortress_Home": { "name": "🏰 魔鬼主場", "roi": -0.008 },
            "Title_MustWin_Home": { "name": "🏆 爭冠必勝盤", "roi": -0.063 },
            "MarketHype_Fav": { "name": "🔥 大熱倒灶", "roi": -0.080 },
            "MidTable_Standard": { "name": "😐 中游例行", "roi": 0.000 }
        }

    # [FIXED] 接收 odds 字典，避免 AttributeError
    def analyze_scenario(self, lh: float, la: float, odds: Dict) -> str:
        home_odd = odds.get("home", 2.0)
        if home_odd < 1.30: return "MarketHype_Fav"
        if (lh + la) < 2.2: return "Bore_Draw_Stalemate"
        if home_odd < 2.0: return "Fortress_Home"
        return "MidTable_Standard"

    def recall_experience(self, regime_id: str) -> Dict:
        return self.history_db.get(regime_id, {"name": "未知", "roi": 0.0})

    def calc_memory_penalty(self, historical_roi: float) -> float:
        if historical_roi < -0.05: return 0.7
        if historical_roi > 0.05: return 1.1
        return 1.0

# =========================
# 3. 分析引擎邏輯 (V37.0 Omni-Engine)
# =========================
class SniperAnalystLogic:
    def __init__(self, json_data: Any, max_g: int = 9, nb_alpha: float = 0.12, lam3: float = 0.0, rho: float = -0.13):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha
        self.lam3 = lam3 
        self.rho = rho 
        self.memory = RegimeMemory()

    def calc_lambda(self) -> Tuple[float, float, bool]:
        """計算 Lambda (含近況加權)"""
        league_base = 1.35
        is_weighted = False
        def att_def_w(team):
            nonlocal is_weighted
            xg, xga = team["offensive_stats"].get("xg_avg", 1.0), team["defensive_stats"].get("xga_avg", 1.0)
            trend = team["context_modifiers"].get("recent_form_trend", [0, 0, 0])
            if any(t != 0 for t in trend): is_weighted = True
            w = np.array([0.1, 0.3, 0.6])
            form_factor = 1.0 + (np.dot(trend[-len(w):], w[-len(trend):]) * 0.1)
            return (0.3 * team["offensive_stats"]["goals_scored_avg"] + 0.7 * xg) * form_factor, \
                   (0.3 * team["defensive_stats"]["goals_conceded_avg"] + 0.7 * xga)

        lh_att, lh_def = att_def_w(self.h)
        la_att, la_def = att_def_w(self.a)
        
        strength_gap = (lh_att - la_att)
        crush_factor = 1.05 if strength_gap > 0.5 else 1.0
        
        if self.h["context_modifiers"].get("missing_key_defender"): lh_def *= 1.25
        if self.a["context_modifiers"].get("missing_key_defender"): la_def *= 1.20
        h_adv = self.h["general_strength"].get("home_advantage_weight", 1.15)
        
        return (lh_att * la_def / league_base) * h_adv * crush_factor, \
               (la_att * lh_def / league_base), is_weighted

    def build_matrix_v37(self, lh: float, la: float, use_biv: bool = True, use_dc: bool = True) -> Tuple[np.ndarray, Dict]:
        """[V37] 全能矩陣生成 (Log-Space Bivariate + Dixon-Coles)"""
        G = self.max_g
        M_model = np.zeros((G, G), dtype=float)
        
        # 1. 物理層 (Log-Space 計算)
        eff_lam3 = max(self.lam3, 0.001) if use_biv else 0.0
        l1 = max(0.01, lh - eff_lam3)
        l2 = max(0.01, la - eff_lam3)
        
        for i in range(G):
            for j in range(G):
                log_p = biv_poisson_logpmf(i, j, l1, l2, eff_lam3)
                M_model[i, j] = math.exp(log_p)

        # 2. Dixon-Coles 修正
        if use_dc:
            def tau(x, y, mu_h, mu_a, rho):
                if x==0 and y==0: return 1.0 - (mu_h * mu_a * rho)
                elif x==0 and y==1: return 1.0 + (mu_h * rho)
                elif x==1 and y==0: return 1.0 + (mu_a * rho)
                elif x==1 and y==1: return 1.0 - rho
                else: return 1.0
            
            for i in range(2):
                for j in range(2):
                    M_model[i, j] *= tau(i, j, lh, la, self.rho)

        M_model /= M_model.sum()

        # 3. 市場混合層
        true_imp = get_true_implied_prob(self.market["1x2_odds"])
        p_h = float(np.sum(np.tril(M_model, -1)))
        p_d = float(np.sum(np.diag(M_model)))
        p_a = float(np.sum(np.triu(M_model, 1)))
        
        market_diff = abs(p_h - true_imp["home"])
        w = 0.7 if market_diff < 0.2 else 0.5 
        
        t_h = w*p_h + (1-w)*true_imp["home"]
        t_d = w*p_d + (1-w)*true_imp["draw"]
        t_a = w*p_a + (1-w)*true_imp["away"]
        
        M_hybrid = M_model.copy()
        M_hybrid[np.tril_indices(G, -1)] *= (t_h / p_h if p_h > 0 else 1)
        M_hybrid[np.diag_indices(G)] *= (t_d / p_d if p_d > 0 else 1)
        M_hybrid[np.triu_indices(G, 1)] *= (t_a / p_a if p_a > 0 else 1)
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
        M_stress = get_matrix_cached(lh, la + 0.3, self.max_g, self.nb_alpha)
        p_orig = float(np.sum(np.tril(get_matrix_cached(lh, la, self.max_g, self.nb_alpha), -1)))
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

    def run_monte_carlo_vectorized(self, M: np.ndarray, sims: int = 100000) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """[V37] 向量化蒙地卡羅"""
        rng = np.random.default_rng()
        flat_probs = M.flatten()
        flat_probs /= flat_probs.sum()
        
        cdf = np.cumsum(flat_probs)
        draws = rng.random(sims)
        indices = np.searchsorted(cdf, draws)
        
        G = M.shape[0]
        home_goals = indices // G
        away_goals = indices % G
        
        h_wins = np.sum(home_goals > away_goals) / sims
        draws = np.sum(home_goals == away_goals) / sims
        a_wins = np.sum(home_goals < away_goals) / sims
        
        return h_wins, draws, a_wins, home_goals, away_goals

    def importance_sampling_over(self, M: np.ndarray, line: float, n_sims: int = 20000) -> Dict[str, Any]:
        """[V37] 向量化重要性採樣"""
        rng = np.random.default_rng()
        G = M.shape[0]; flat = M.flatten()
        idx = np.arange(G*G); i = idx // G; j = idx % G
        sums = (i + j).astype(float)
        bias = (1.0 + sums) ** 1.5 
        q = flat * bias; q /= q.sum()
        
        draws_idx = rng.choice(G*G, size=n_sims, p=q)
        weights = flat[draws_idx] / q[draws_idx]
        
        indicators = (sums[draws_idx] > line)
        est = np.sum(weights * indicators) / np.sum(weights)
        return {"est": float(est)}

# =========================
# 4. MLE 參數自動擬合
# =========================
def fit_params_mle(history_df: pd.DataFrame) -> Dict[str, float]:
    def neg_log_likelihood(params):
        lam3, rho = params
        nll = 0.0
        if not (0 <= lam3 <= 0.5) or not (-0.2 <= rho <= 0.2): return 1e9
        for _, row in history_df.iterrows():
            try:
                lh, la = float(row['lh_pred']), float(row['la_pred'])
                h, a = int(row['home_goals']), int(row['away_goals'])
                l1 = max(0.01, lh - lam3)
                l2 = max(0.01, la - lam3)
                lp = biv_poisson_logpmf(h, a, l1, l2, lam3)
                prob = math.exp(lp)
                if h==0 and a==0: prob *= (1 - lh*la*rho)
                elif h==0 and a==1: prob *= (1 + lh*rho)
                elif h==1 and a==0: prob *= (1 + la*rho)
                elif h==1 and a==1: prob *= (1 - rho)
                nll -= math.log(max(prob, 1e-9))
            except: continue
        return nll
    initial_guess = [0.1, -0.1]
    result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead', tol=1e-3)
    return {"lam3": result.x[0], "rho": result.x[1], "success": result.success}

# =========================
# 5. Streamlit UI (V37.2 Omni-Analyst)
# =========================
st.set_page_config(page_title="Sniper V37.2 Omni-Analyst", page_icon="🧿", layout="wide")

st.markdown("""
<style>
    .metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧿 Sniper V37.2")
    st.caption("Omni-Analyst Edition")
    st.markdown("---")
    app_mode = st.radio("功能模式：", ["🎯 單場深度預測", "🛡️ 風險對沖實驗室", "🔧 參數校正實驗室", "📈 聯賽歷史回測", "📚 劇本查詢"])
    st.divider()
    with st.expander("🛠️ 進階參數", expanded=False):
        unit_stake = st.number_input("單注本金 ($)", 10, 10000, 100)
        nb_alpha = st.slider("Alpha (NB)", 0.05, 0.25, 0.12)
        use_biv = st.toggle("啟用 Bivariate Poisson", value=True)
        use_dc = st.toggle("啟用 Dixon-Coles", value=True)
        c1, c2 = st.columns(2)
        lam3_input = c1.number_input("Lambda 3", 0.0, 0.5, 0.15, step=0.01)
        rho_input = c2.number_input("Rho (DC)", -0.3, 0.3, -0.13, step=0.01)
        risk_scale = st.slider("風險係數", 0.1, 1.0, 0.3)
        use_mock_memory = st.checkbox("歷史記憶修正", value=True)
        show_uncertainty = st.toggle("顯示 EV 不確定區間", value=True)

# =========================
# 模式 1: 單場深度預測
# =========================
if app_mode == "🎯 單場深度預測":
    st.header("🎯 單場深度預測 (V37 Engine)")
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

    if st.button("🚀 執行高速分析", type="primary"):
        if input_data:
            engine = SniperAnalystLogic(input_data, 9, nb_alpha, lam3_input, rho_input)
            lh, la, is_weighted = engine.calc_lambda()
            M, probs_detail = engine.build_matrix_v37(lh, la, use_biv=use_biv, use_dc=use_dc)
            market_bonus = engine.get_market_trend_bonus()
            
            # [FIXED] 參數傳遞修正
            odds_dict = engine.market["1x2_odds"]
            regime_id = engine.memory.analyze_scenario(lh, la, odds_dict)
            history_data = engine.memory.recall_experience(regime_id)
            penalty = engine.memory.calc_memory_penalty(history_data["roi"]) if use_mock_memory else 1.0
            
            p_h = probs_detail["hybrid"]["home"]
            m_h = probs_detail["market"]["home"]
            diff_p = abs(p_h - m_h) / max(m_h, 1e-9)
            sens_lv, sens_dr = engine.check_sensitivity(lh, la)
            conf_score, conf_reasons = engine.calc_model_confidence(lh, la, diff_p, sens_dr)
            
            # 預先計算 MC
            hw, dr, aw, sh, sa = engine.run_monte_carlo_vectorized(M, sims=100000)

            st.session_state.analysis_results = {
                "engine": engine, "M": M, "lh": lh, "la": la, "is_weighted": is_weighted,
                "probs_detail": probs_detail, "market_bonus": market_bonus,
                "history_data": history_data, "penalty": penalty,
                "conf_score": conf_score, "conf_reasons": conf_reasons,
                "sim_data": {"sh": sh, "sa": sa} 
            }
        else:
            st.error("請輸入數據")

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        engine, M = res["engine"], res["M"]
        probs = res["probs_detail"]

        st.markdown("### 🔍 V37 戰術儀表板")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("主隊進球預期", f"{res['lh']:.2f}", delta="加權啟用" if res["is_weighted"] else None)
        d2.metric("客隊進球預期", f"{res['la']:.2f}")
        hybrid_p = probs["hybrid"]["home"]
        model_p = probs["model"]["home"]
        d3.metric("V37 混合主勝", f"{hybrid_p:.1%}", delta=f"{(hybrid_p - model_p)*100:+.1f}%", delta_color="inverse")
        conf_score = res["conf_score"]
        d4.metric("🛡️ 信心指數", f"{conf_score:.0%}")
        
        if conf_score < 1.0:
            with st.expander(f"⚠️ 信心扣分診斷", expanded=True):
                for r in res["conf_reasons"]: st.warning(f"🔻 {r}")

        t_val, t_ai, t_score, t_sim = st.tabs(["💰 價值投資", "🧠 智能裁決", "🎯 波膽分佈", "🎲 高速模擬"])
        candidates = [] 
        
        with t_val:
            st.subheader("獨贏 (1x2)")
            r_1x2 = []
            for tag, key in [("主勝", "home"), ("和局", "draw"), ("客勝", "away")]:
                prob = probs["hybrid"][key]
                odd = engine.market["1x2_odds"][key]
                raw_ev = (prob * odd - 1) * 100 + res["market_bonus"][key]
                adj_ev = raw_ev * conf_score * res["penalty"]
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
            st.write("V37 權重混合分析")
            df_comp = pd.DataFrame([probs["model"], probs["market"], probs["hybrid"]], index=["純模型", "市場去水", "V37混合"])
            st.dataframe(df_comp.style.format("{:.1%}"))

        with t_score:
            st.write("波膽矩陣 (Hybrid)")
            st.dataframe(pd.DataFrame(M[:6,:6]).style.format("{:.1%}"))

        with t_sim:
            st.subheader("高速蒙地卡羅 (100,000 runs)")
            sh, sa = res["sim_data"]["sh"], res["sim_data"]["sa"]
            hw = np.sum(sh > sa) / 100000
            dr = np.sum(sh == sa) / 100000
            aw = np.sum(sh < sa) / 100000
            
            c1, c2, c3 = st.columns(3)
            c1.metric("主勝率 (MC)", f"{hw:.1%}")
            c2.metric("和局率 (MC)", f"{dr:.1%}")
            c3.metric("客勝率 (MC)", f"{aw:.1%}")
            
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(sh, alpha=0.5, label="Home", bins=range(8), density=True)
            ax.hist(sa, alpha=0.5, label="Away", bins=range(8), density=True)
            ax.legend(); st.pyplot(fig)
            
            st.divider()
            st.subheader("稀有事件 (Importance Sampling)")
            line_check = 4.5
            is_res = engine.importance_sampling_over(M, line_check)
            st.metric(f"大 {line_check} 機率", f"{is_res['est']:.2%}")

# =========================
# 模式 2: 風險對沖實驗室 (連動 + 智能評語)
# =========================
elif app_mode == "🛡️ 風險對沖實驗室":
    st.title("🛡️ 風險對沖實驗室 (Hedging Lab)")
    st.markdown("提供 **套利檢測**、**Lay 對沖** 與 **投資組合優化** 工具。")
    
    has_data = "analysis_results" in st.session_state and st.session_state.analysis_results is not None
    
    tab_arb, tab_lay, tab_port = st.tabs(["⚡ 套利計算機", "📉 Lay 對沖", "📊 組合優化"])
    
    with tab_arb:
        st.subheader("1x2 套利掃描")
        c1, c2, c3 = st.columns(3)
        def_o = st.session_state.analysis_results["engine"].market["1x2_odds"] if has_data else {"home":2.0, "draw":3.0, "away":4.0}
        
        o_h = c1.number_input("主勝賠率", 1.01, 100.0, def_o["home"])
        o_d = c2.number_input("和局賠率", 1.01, 100.0, def_o["draw"])
        o_a = c3.number_input("客勝賠率", 1.01, 100.0, def_o["away"])
        
        inv_sum = (1/o_h) + (1/o_d) + (1/o_a)
        arb_ret = (1/inv_sum - 1) * 100
        
        if inv_sum < 1.0:
            st.success(f"🔥 發現套利機會！理論利潤: **{arb_ret:.2f}%**")
            target_profit = st.number_input("目標總利潤 ($)", 100, 10000, 1000)
            st.write("建議下注額 (Dutching):")
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("主勝下注", f"${target_profit/(inv_sum*o_h):.0f}")
            c_s2.metric("和局下注", f"${target_profit/(inv_sum*o_d):.0f}")
            c_s3.metric("客勝下注", f"${target_profit/(inv_sum*o_a):.0f}")
        else:
            st.info(f"無套利空間 (Book Sum: {inv_sum:.2%})")

    with tab_lay:
        st.subheader("交易所對沖計算器 (Back-Lay)")
        lc1, lc2 = st.columns(2)
        back_odds = lc1.number_input("Back 賠率 (Bookie)", 1.01, 100.0, 2.5)
        back_stake = lc1.number_input("Back 本金 ($)", 10, 10000, 100)
        lay_odds = lc2.number_input("Lay 賠率 (Exchange)", 1.01, 100.0, 2.4)
        comm = lc2.number_input("佣金 (%)", 0.0, 10.0, 2.0) / 100.0
        
        if lay_odds > 1.0:
            lay_stake = (back_stake * back_odds) / (lay_odds - comm)
            liability = lay_stake * (lay_odds - 1)
            profit = (back_odds - 1)*back_stake - (lay_odds - 1)*lay_stake
            
            st.metric("建議 Lay 金額", f"${lay_stake:.2f}")
            st.write(f"需預留負債: **${liability:.2f}** | 鎖定利潤: **${profit:.2f}**")

    # 3. 組合優化 (加入智能評語)
    with tab_port:
        st.subheader("智能組合優化 (Portfolio Optimization)")
        if has_data:
            res = st.session_state.analysis_results
            sh, sa = res["sim_data"]["sh"], res["sim_data"]["sa"]
            engine = res["engine"]
            
            st.info("已載入單場預測的 100,000 次模擬數據。")
            
            candidates = [
                {"name": "主勝", "odds": engine.market["1x2_odds"]["home"], "cond": (sh > sa)},
                {"name": "和局", "odds": engine.market["1x2_odds"]["draw"], "cond": (sh == sa)},
                {"name": "客勝", "odds": engine.market["1x2_odds"]["away"], "cond": (sh < sa)},
                {"name": "大 2.5", "odds": engine.market.get("target_odds", 1.9), "cond": ((sh+sa) > 2.5)},
                {"name": "小 2.5", "odds": engine.market.get("target_odds", 1.9), "cond": ((sh+sa) < 2.5)}
            ]

            if st.button("⚡ 計算最佳資金分配 (Markowitz)"):
                payoffs = np.zeros((100000, len(candidates)))
                for i, c in enumerate(candidates):
                    payoffs[:, i] = np.where(c["cond"], c["odds"] - 1, -1)
                
                mu = np.mean(payoffs, axis=0)
                sigma = np.cov(payoffs, rowvar=False)
                
                n = len(candidates)
                def objective(w):
                    ret = np.dot(w, mu)
                    risk = np.dot(w.T, np.dot(sigma, w))
                    return -(ret - 0.5 * 2.0 * risk) # Gamma=2.0
                
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bnds = tuple((0, 1) for _ in range(n))
                init_guess = [1/n] * n
                
                try:
                    res_opt = minimize(objective, init_guess, bounds=bnds, constraints=cons)
                    weights = res_opt.x
                    
                    st.write("**最佳權重分配 (Risk Aversion = 2.0):**")
                    cols = st.columns(n)
                    active_bets = []
                    for i, w in enumerate(weights):
                        if w > 0.05:
                            active_bets.append((candidates[i]["name"], w))
                            cols[i].metric(candidates[i]["name"], f"{w:.1%}", delta=f"Exp. Ret: {mu[i]*100:.1f}%")
                        else:
                            cols[i].metric(candidates[i]["name"], "0.0%", delta_color="off")
                    
                    # --- 👨‍🏫 首席分析師總結 ---
                    st.divider()
                    st.markdown("### 👨‍🏫 首席分析師評語 (Verdict)")
                    
                    max_w = max(weights)
                    top_pick = max(active_bets, key=lambda x: x[1])[0] if active_bets else "無"
                    total_exp_return = np.dot(weights, mu) * 100
                    
                    verdict_color = "blue"
                    verdict_title = "觀察"
                    verdict_text = ""

                    if not active_bets or total_exp_return < 0.5:
                        verdict_color = "red"
                        verdict_title = "⛔ 風險過高 / 無價值"
                        verdict_text = "Sniper 模型經運算後認為，此場比賽**無任何注單具備足夠的風險回報比**。即使分散投資，期望值依然過低。建議**直接跳過此場**，保留資金。"
                    
                    elif max_w > 0.7:
                        verdict_color = "green"
                        verdict_title = "🔥 強力單注出擊"
                        verdict_text = f"模型對 **【{top_pick}】** 展現出極高的信心 (權重 > 70%)。這表示模擬結果顯示該選項與其他選項的關聯風險極低。建議**集中資金單打**此選項，無需過度對沖。"
                    
                    elif len(active_bets) >= 2:
                        verdict_color = "orange"
                        verdict_title = "⚖️ 結構化對沖組合"
                        picks_str = " + ".join([f"{p[0]}" for p in active_bets])
                        verdict_text = f"模型建議採取 **「組合拳」** 策略。主要由 **【{picks_str}】** 構成。這代表這些選項在數學上具有**互補性** (例如：主勝通常伴隨大分)。請務必**依照建議比例分注**，才能有效降低單邊倒的風險。"
                    
                    else:
                        verdict_color = "blue"
                        verdict_title = "🔵 一般價值投資"
                        verdict_text = f"發現些微價值，主要集中在 {top_pick}，但優勢並非壓倒性。建議小注怡情。"

                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 5px; border-left: 5px solid {verdict_color}; background-color: #f0f2f6;">
                        <h4 style="margin:0; color:{verdict_color};">{verdict_title}</h4>
                        <p style="margin-top:10px; font-size:16px;">{verdict_text}</p>
                        <hr>
                        <small>📊 組合預期回報率 (Portfolio EV): <b>{total_exp_return:.2f}%</b></small>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"優化失敗: {e}")
        else:
            st.warning("⚠️ 請先在「單場深度預測」執行分析，以生成模擬數據。")

# =========================
# 模式 3: 參數校正實驗室
# =========================
elif app_mode == "🔧 參數校正實驗室":
    st.header("🔧 參數校正實驗室 (Auto-Calibration)")
    st.markdown("利用 `scipy.optimize` 尋找歷史數據中的最佳 Lambda3 (共變異) 與 Rho (DC校正)")
    
    cal_file = st.file_uploader("上傳含有 lh_pred, la_pred, home_goals, away_goals 的 CSV", type=['csv'])
    
    if cal_file:
        df_cal = pd.read_csv(cal_file)
        st.write("預覽數據:", df_cal.head())
        if st.button("⚡ 開始 MLE 擬合", type="primary"):
            with st.spinner("正在進行最大概似估計 (MLE)..."):
                best_params = fit_params_mle(df_cal)
                
            if best_params["success"]:
                st.success("校正成功！請將以下參數填入側邊欄：")
                c1, c2 = st.columns(2)
                c1.metric("最佳 Lambda3", f"{best_params['lam3']:.3f}")
                c2.metric("最佳 Rho (DC)", f"{best_params['rho']:.3f}")
            else:
                st.error("校正收斂失敗，請檢查數據品質。")
    else:
        st.info("無數據時，可生成模擬數據進行測試。")
        if st.button("生成模擬測試數據"):
            mock_df = pd.DataFrame({
                'lh_pred': np.random.uniform(1.0, 2.5, 100),
                'la_pred': np.random.uniform(0.8, 2.0, 100),
                'home_goals': np.random.randint(0, 5, 100),
                'away_goals': np.random.randint(0, 4, 100)
            })
            st.dataframe(mock_df)
            st.caption("請將此表格複製並存為 CSV 上傳。")

# =========================
# 模式 4 & 5
# =========================
elif app_mode == "📈 聯賽歷史回測":
    st.title("📈 聯賽歷史回測")
    st.info("請將 CSV 檔案放入資料夾後，使用 V37 Batch Engine 進行測試。")

elif app_mode == "📚 劇本查詢":
    st.title("📚 盤口劇本庫")
    mem = RegimeMemory()
    data = [{"劇本": v["name"], "ROI": f"{v['roi']:.1%}", "樣本": v["bets"]} for k, v in mem.history_db.items()]
    st.dataframe(pd.DataFrame(data), use_container_width=True)
