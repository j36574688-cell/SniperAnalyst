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
# 1. 核心數學工具
# =========================

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def nb_pmf(k: int, mu: float, alpha: float) -> float:
    if alpha <= 0: return poisson_pmf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    coeff = math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1))
    return float(coeff * (p ** r) * ((1 - p) ** k))

def get_true_implied_prob(odds_dict: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / float(v) if v > 0 else 0.0 for k, v in odds_dict.items()}
    margin = sum(inv.values())
    return {k: inv[k] / margin if margin > 0 else 0.0 for k in odds_dict}

@st.cache_data
def get_matrix_cached(lh: float, la: float, max_g: int, nb_alpha: float, vol_adjust: bool) -> np.ndarray:
    G = max_g
    i, j = np.arange(G), np.arange(G)
    p_i = np.array([poisson_pmf(k, lh) for k in i]); p_j = np.array([poisson_pmf(k, la) for k in j])
    Mp = np.outer(p_i, p_j)
    nb_i = np.array([nb_pmf(k, lh, nb_alpha) for k in i]); nb_j = np.array([nb_pmf(k, la, nb_alpha) for k in j])
    Mn = np.outer(nb_i, nb_j)
    M = 0.6 * Mp + 0.4 * Mn
    rho = -0.18 if vol_adjust else -0.13
    if G > 1:
        M[0,0] *= (1 - lh*la*rho); M[1,0] *= (1 + la*rho)
        M[0,1] *= (1 + lh*rho); M[1,1] *= (1 - rho)
    return M / M.sum()

def calc_risk_adj_kelly(ev_percent: float, variance: float, risk_scale: float = 0.5, prob: float = 0.5) -> float:
    if variance <= 0 or ev_percent <= 0: return 0.0
    ev = ev_percent / 100.0
    f = (ev / variance) * risk_scale
    cap = 0.5 if prob >= 0.35 else 0.02
    return min(cap, max(0.0, f)) * 100

def calc_risk_metrics(prob: float, odds: float) -> Tuple[float, float]:
    if prob <= 0 or prob >= 1: return 0.0, 0.0
    win_p, lose_p = odds - 1.0, -1.0
    ev = prob * win_p + (1 - prob) * lose_p
    var = prob * (win_p**2) + (1 - prob) * (lose_p**2) - (ev**2)
    sharpe = ev / math.sqrt(var) if var > 0 else 0
    return var, sharpe

# =========================
# 2. 全景記憶體系
# =========================
class RegimeMemory:
    def __init__(self):
        self.history_db = {
            "Bore_Draw_Stalemate": { "name": "🛡️ 雙重鐵桶 (悶和局)", "bets": 19, "roi": 0.219 }, 
            "Relegation_Dog": { "name": "🐕 保級受讓 (絕境爆發)", "bets": 101, "roi": 0.083 },
            "Fallen_Giant": { "name": "📉 豪門崩盤 (名氣大狀況差)", "bets": 67, "roi": -0.008 },
            "Fortress_Home": { "name": "🏰 魔鬼主場 (主場過熱)", "bets": 256, "roi": -0.008 },
            "Counter_Away_Dog": { "name": "⚡ 客隊防反 (偷襲得手)", "bets": 90, "roi": 0.010 },
            "MidTable_Standard": { "name": "😐 中游例行公事", "bets": 300, "roi": 0.000 },
            "Title_MustWin_Home": { "name": "🏆 爭冠必勝盤 (溢價陷阱)", "bets": 256, "roi": -0.063 },
            "Injury_Crisis_Fav": { "name": "🏥 傷兵詛咒 (無力回天)", "bets": 37, "roi": -0.099 },
            "Hidden_Gem_Dog": { "name": "🦊 扮豬吃老虎 (數據失靈)", "bets": 6, "roi": -0.117 },
            "MarketHype_Fav": { "name": "🔥 大熱倒灶 (過度熱門)", "bets": 150, "roi": -0.080 },
            "HeavyFav_DeepBlock": { "name": "⚠️ 強隊遇鐵桶陣", "bets": 50, "roi": -0.120 }
        }

    def analyze_scenario(self, engine: 'SniperAnalystLogic', lh: float, la: float) -> str:
        h, a = engine.h, engine.a
        odds = engine.market["1x2_odds"]
        prob_h = 1.0 / odds["home"]
        h_odds = odds["home"]
        form_h_score = sum(h["context_modifiers"].get("recent_form_trend", [0]))
        motiv_h = h["context_modifiers"]["motivation"]
        
        if odds["home"] < 2.10 and form_h_score < -1: return "Fallen_Giant"
        if prob_h > 0.65 and form_h_score < 0: return "Injury_Crisis_Fav"
        if motiv_h == "title_race" and prob_h > 0.65: return "Title_MustWin_Home"
        if odds["home"] < 1.30: return "MarketHype_Fav"
        if (lh + la) < 2.2: return "Bore_Draw_Stalemate"
        return "MidTable_Standard"

    def recall_experience(self, regime_id: str) -> Dict:
        return self.history_db.get(regime_id, {"name": "🔍 未知盤口", "bets": 0, "roi": 0.0})

    def calc_memory_penalty(self, historical_roi: float) -> float:
        if historical_roi < -0.10: return 0.5
        if historical_roi < -0.05: return 0.7
        if historical_roi > 0.05: return 1.1
        return 1.0

# =========================
# 3. 分析引擎邏輯
# =========================
class SniperAnalystLogic:
    def __init__(self, json_data: Any, max_g: int = 9, nb_alpha: float = 0.12):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha
        self.memory = RegimeMemory()

    def calc_lambda(self) -> Tuple[float, float, bool]:
        """[V33] 加入 Time-Decay 近況加權"""
        league_base = 1.35
        is_weighted = False
        def att_def_w(team):
            nonlocal is_weighted
            xg, xga = team["offensive_stats"].get("xg_avg", team["offensive_stats"]["goals_scored_avg"]), team["defensive_stats"].get("xga_avg", team["defensive_stats"]["goals_conceded_avg"])
            trend = team["context_modifiers"].get("recent_form_trend", [0, 0, 0])
            if any(trend): is_weighted = True
            w = np.array([0.1, 0.3, 0.6])
            form_factor = 1.0 + (np.dot(trend[-len(w):], w[-len(trend):]) * 0.1)
            return (0.3 * team["offensive_stats"]["goals_scored_avg"] + 0.7 * xg) * form_factor, (0.3 * team["defensive_stats"]["goals_conceded_avg"] + 0.7 * xga)

        lh_att, lh_def = att_def_w(self.h)
        la_att, la_def = att_def_w(self.a)
        if self.h["context_modifiers"].get("missing_key_defender"): lh_def *= 1.25
        if self.a["context_modifiers"].get("missing_key_defender"): la_def *= 1.20
        h_adv = self.h["general_strength"].get("home_advantage_weight", 1.15)
        return (lh_att * la_def / league_base) * h_adv, (la_att * lh_def / league_base), is_weighted

    def build_ensemble_matrix(self, lh: float, la: float) -> Tuple[np.ndarray, Dict]:
        """[V33] 混合矩陣 (回傳矩陣與機率細節)"""
        vol_adjust = (self.h.get("style_of_play", {}).get("volatility") == "high")
        M_model = get_matrix_cached(lh, la, self.max_g, self.nb_alpha, vol_adjust)
        true_imp = get_true_implied_prob(self.market["1x2_odds"])
        
        # 1. 物理模型機率
        p_h, p_d, p_a = float(np.sum(np.tril(M_model, -1))), float(np.sum(np.diag(M_model))), float(np.sum(np.triu(M_model, 1)))
        
        # 2. 混合疊加權重: Model 70% + Market 30%
        w = 0.7
        t_h, t_d, t_a = w*p_h + (1-w)*true_imp["home"], w*p_d + (1-w)*true_imp["draw"], w*p_a + (1-w)*true_imp["away"]
        
        # 3. 矩陣再平衡
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

    def run_monte_carlo(self, M: np.ndarray, sims: int = 10000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        rng = np.random.default_rng(seed); G = M.shape[0]; flat_M = M.flatten()
        indices = rng.choice(G * G, size=sims, p=flat_M/flat_M.sum())
        sh, sa = indices // G, indices % G
        res = np.full(sims, "draw", dtype=object); res[sh > sa] = "home"; res[sh < sa] = "away"
        return sh, sa, res.tolist()

    def check_sensitivity(self, lh: float, la: float) -> Tuple[str, float]:
        M_stress = get_matrix_cached(lh, la + 0.3, self.max_g, self.nb_alpha, False)
        # 用純物理矩陣做敏感度測試
        p_i = np.array([poisson_pmf(k, lh) for k in range(self.max_g)])
        p_j = np.array([poisson_pmf(k, la) for k in range(self.max_g)])
        M_orig = np.outer(p_i, p_j); M_orig /= M_orig.sum()
        p_h_orig = float(np.sum(np.tril(M_orig, -1)))
        p_h_new = float(np.sum(np.tril(M_stress, -1)))
        drop = (p_h_orig - p_h_new) / p_h_orig if p_h_orig > 0 else 0
        return ("High" if drop > 0.15 else "Medium" if drop > 0.08 else "Low"), drop

    def calc_model_confidence(self, lh: float, la: float, market_diff: float, sens_drop: float) -> Tuple[float, List[str]]:
        score, reasons = 1.0, []
        if market_diff > 0.25: score *= 0.7; reasons.append(f"與市場差異過大 ({market_diff:.1%})")
        if sens_drop > 0.15: score *= 0.8; reasons.append("模型對運氣球極度敏感")
        return score, reasons

# =========================
# 4. Streamlit UI
# =========================
st.set_page_config(page_title="Sniper V33.0 Ultimate", page_icon="🎯", layout="wide")

# CSS 優化：讓儀表板更好看
st.markdown("""
<style>
    .metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; }
    .highlight { color: #ff4b4b; font-weight: bold; }
    .reco-card { border-left: 5px solid #28a745; background-color: #e6ffed; padding: 10px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.title("🎯 Sniper V33.0")
    st.markdown("---")
    app_mode = st.radio("功能模式：", ["🎯 單場深度預測 (V32介面)", "📈 聯賽歷史回測", "📚 劇本與 ROI 查詢"])
    st.divider()
    with st.expander("🛠️ 進階參數 (V33核心)", expanded=False):
        unit_stake = st.number_input("單注本金 ($)", 10, 10000, 100)
        nb_alpha = st.slider("Alpha (變異數)", 0.05, 0.20, 0.12)
        risk_scale = st.slider("風險係數", 0.1, 1.0, 0.3)
        use_mock_memory = st.checkbox("歷史記憶修正", value=True)

# =========================
# 模式 1: 單場深度預測
# =========================
if app_mode == "🎯 單場深度預測 (V32介面)":
    st.header("🎯 單場深度預測 (V33 Hybrid Core)")
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

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

    if st.button("🚀 執行 V33 分析", type="primary"):
        if input_data:
            engine = SniperAnalystLogic(input_data, 9, nb_alpha)
            # 1. 計算 (包含是否加權的 flag)
            lh, la, is_weighted = engine.calc_lambda()
            # 2. 矩陣與機率細節
            M, probs_detail = engine.build_ensemble_matrix(lh, la)
            
            market_bonus = engine.get_market_trend_bonus()
            regime_id = engine.memory.analyze_scenario(engine, lh, la)
            history_data = engine.memory.recall_experience(regime_id)
            penalty = engine.memory.calc_memory_penalty(history_data["roi"]) if use_mock_memory else 1.0
            
            # 信心度計算 (用混合後的機率 vs 市場)
            p_h = probs_detail["hybrid"]["home"]
            m_h = probs_detail["market"]["home"]
            diff_p = abs(p_h - m_h) / max(m_h, 1e-9)
            sens_lv, sens_dr = engine.check_sensitivity(lh, la)
            conf_score, conf_reasons = engine.calc_model_confidence(lh, la, diff_p, sens_dr)

            st.session_state.analysis_results = {
                "engine": engine, "M": M, "lh": lh, "la": la, "is_weighted": is_weighted,
                "probs_detail": probs_detail, "market_bonus": market_bonus,
                "history_data": history_data, "penalty": penalty,
                "conf_score": conf_score, "prob_h": p_h
            }
        else:
            st.error("請輸入數據")

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        engine, M = res["engine"], res["M"]
        probs = res["probs_detail"]

        # --- [V33 視覺增強] 差異儀表板 ---
        st.markdown("### 🔍 V33 差異監測器 (Diff Monitor)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("主隊進球預期", f"{res['lh']:.2f}", delta="近況加權" if res["is_weighted"] else None)
        d2.metric("客隊進球預期", f"{res['la']:.2f}")
        
        # 顯示 V32 (純模型) vs V33 (混合) 的機率差異
        model_p = probs["model"]["home"]
        hybrid_p = probs["hybrid"]["home"]
        delta_p = (hybrid_p - model_p) * 100
        d3.metric("主勝機率 (V33)", f"{hybrid_p:.1%}", delta=f"{delta_p:+.1f}% (修正)", delta_color="inverse")
        d4.metric("市場隱含機率", f"{probs['market']['home']:.1%}")
        
        # --- 原始 V32 側邊欄 ---
        with st.sidebar:
            st.divider(); st.subheader("盤口劇本"); st.info(res['history_data']['name'])
            if use_mock_memory: st.metric("歷史 ROI", f"{res['history_data']['roi']:.1%}")
            st.metric("模型信心", f"{res['conf_score']:.0%}")

        # --- 內容 Tabs ---
        t_val, t_ai, t_score, t_sim = st.tabs(["💰 價值投資建議", "🧠 智能裁決", "🎯 波膽分佈", "🎲 模擬"])
        
        candidates = [] # 收集所有注單
        
        with t_val:
            # 1. 1x2 獨贏
            st.subheader("獨贏 (1x2)")
            r_1x2 = []
            for tag, key in [("主勝", "home"), ("和局", "draw"), ("客勝", "away")]:
                prob = probs["hybrid"][key]
                odd = engine.market["1x2_odds"][key]
                raw_ev = (prob * odd - 1) * 100 + res["market_bonus"][key]
                adj_ev = raw_ev * res["conf_score"] * res["penalty"]
                var, sharpe = calc_risk_metrics(prob, odd)
                kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, prob)
                r_1x2.append({"選項": tag, "賠率": odd, "修正EV": f"{adj_ev:+.1f}%", "注碼%": f"{kelly:.1f}%"})
                if adj_ev > 1.0: candidates.append({"pick": tag, "odds": odd, "ev": adj_ev, "kelly": kelly, "type": "1x2", "sharpe": sharpe})
            st.dataframe(pd.DataFrame(r_1x2), use_container_width=True)
            
            # 2. 亞盤 AH
            c_ah, c_ou = st.columns(2)
            with c_ah:
                st.subheader("亞盤 (AH)")
                target_o = engine.market.get("target_odds", 1.90)
                for hcap in engine.market["handicaps"]:
                    raw_ev = engine.ah_ev(M, hcap, target_o) + res["market_bonus"]["home"]
                    adj_ev = raw_ev * res["conf_score"] * res["penalty"]
                    # 反推隱含機率
                    prob_apx = (raw_ev/100.0 + 1) / target_o
                    var, sharpe = calc_risk_metrics(prob_apx, target_o)
                    kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, prob_apx)
                    if adj_ev > 1.5: candidates.append({"pick": f"主 {hcap:+}", "odds": target_o, "ev": adj_ev, "kelly": kelly, "type": "AH", "sharpe": sharpe})
                    st.write(f"主 {hcap:+}: **{adj_ev:+.1f}%** EV")

            with c_ou:
                st.subheader("大小 (OU)")
                idx_sum = np.add.outer(np.arange(engine.max_g), np.arange(engine.max_g))
                for line in engine.market["goal_lines"]:
                    p_over = float(M[idx_sum > line].sum())
                    raw_ev = (p_over * target_o - 1) * 100
                    adj_ev = raw_ev * res["conf_score"] * res["penalty"]
                    var, sharpe = calc_risk_metrics(p_over, target_o)
                    kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, p_over)
                    if adj_ev > 1.5: candidates.append({"pick": f"大 {line}", "odds": target_o, "ev": adj_ev, "kelly": kelly, "type": "OU", "sharpe": sharpe})
                    st.write(f"大 {line}: **{adj_ev:+.1f}%** EV")

            # --- 🔥 智能排行榜回歸 ---
            st.divider()
            st.markdown("### 🏆 智能投資組合 (Smart Portfolio)")
            if candidates:
                # 排序
                best_picks = sorted(candidates, key=lambda x: x['ev'], reverse=True)[:3]
                
                reco_data = []
                for p in best_picks:
                    amt = unit_stake * (p['kelly'] / 100.0)
                    icon = "🟢" if p['sharpe'] > 0.1 else "🟡"
                    reco_data.append([
                        f"[{p['type']}] {p['pick']}", 
                        p['odds'], 
                        f"{p['ev']:+.1f}%", 
                        f"{icon} {p['sharpe']:.2f}",
                        f"{p['kelly']:.1f}%",
                        f"${amt:.1f}"
                    ])
                
                df_reco = pd.DataFrame(reco_data, columns=["選項", "賠率", "修正EV", "夏普值", "注碼%", "建議金額"])
                st.dataframe(df_reco, use_container_width=True)
                
                # 簡單文字總結
                top = best_picks[0]
                st.success(f"🔥 首選推薦：**{top['pick']}** (EV {top['ev']:.1f}%)，建議本金投入 {top['kelly']:.1f}%")
            else:
                st.info("🚧 系統運算後，本場無高價值 (EV > 1.5%) 注單，建議觀望。")

        with t_ai:
            st.write("V33 混合權重分析 (Model vs Market)")
            df_comp = pd.DataFrame([probs["model"], probs["market"], probs["hybrid"]], index=["純模型", "市場去水", "V33混合"])
            st.dataframe(df_comp.style.format("{:.1%}"))

        with t_score:
            st.write("波膽矩陣 (Hybrid Matrix)")
            st.dataframe(pd.DataFrame(M[:6,:6]).style.format("{:.1%}"))

        with t_sim:
            sh, sa, sr = engine.run_monte_carlo(M)
            st.metric("主勝率 (MC)", f"{sr.count('home')/100:.1f}%")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(sh, alpha=0.5, label="Home"); ax.hist(sa, alpha=0.5, label="Away"); ax.legend()
            st.pyplot(fig)

# =========================
# 模式 2 & 3: 佔位符 (功能已預留)
# =========================
elif app_mode == "📈 聯賽歷史回測":
    st.title("📈 聯賽歷史回測")
    st.info("請將 CSV 檔案放入資料夾後，使用 V33 回測引擎進行批量驗證。")
    # (此處可放入之前的 Batch Backtest 邏輯)

elif app_mode == "📚 劇本與 ROI 查詢":
    st.title("📚 盤口劇本庫")
    mem = RegimeMemory()
    data = [{"劇本": v["name"], "ROI": f"{v['roi']:.1%}"} for k, v in mem.history_db.items()]
    st.dataframe(pd.DataFrame(data))
