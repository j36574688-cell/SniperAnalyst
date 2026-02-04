import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# =========================
# 1. 核心數學工具 (Math Utils)
# =========================

def poisson_pmf(k: int, lam: float) -> float:
    """[V33.0] 基礎 Poisson 機率"""
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def nb_pmf(k: int, mu: float, alpha: float) -> float:
    """[V33.0] 負二項分佈 (處理高變異比賽)"""
    if alpha <= 0: return poisson_pmf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    coeff = math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1))
    return float(coeff * (p ** r) * ((1 - p) ** k))

def implied_prob(odds: float) -> float:
    """[V33.0] 計算隱含機率 (倒數)"""
    return 1.0 / odds if odds > 1.0 else 0.0

@st.cache_data
def get_base_matrix(lh: float, la: float, max_g: int, nb_alpha: float, vol_adjust: bool) -> np.ndarray:
    """[V33.0] 生成基礎物理矩陣 (Model Matrix)"""
    G = max_g
    i = np.arange(G)
    j = np.arange(G)
    
    # 混合 Poisson 與 Negative Binomial
    p_i = np.array([poisson_pmf(k, lh) for k in i])
    p_j = np.array([poisson_pmf(k, la) for k in j])
    Mp = np.outer(p_i, p_j)

    nb_i = np.array([nb_pmf(k, lh, nb_alpha) for k in i])
    nb_j = np.array([nb_pmf(k, la, nb_alpha) for k in j])
    Mn = np.outer(nb_i, nb_j)

    # V33 設定：模型權重 60% NB (抗波動), 40% Poisson
    M = 0.4 * Mp + 0.6 * Mn
    
    # 相關性修正 (Dependency Correction)
    rho = -0.18 if vol_adjust else -0.12
    if G > 1:
        M[0,0] *= (1 - lh*la*rho)
        M[1,0] *= (1 + la*rho)
        M[0,1] *= (1 + lh*rho)
        M[1,1] *= (1 - rho)
        
    return M / M.sum()

def calc_risk_adj_kelly(ev_percent: float, variance: float, risk_scale: float = 0.5, prob: float = 0.5) -> float:
    if variance <= 0 or ev_percent <= 0: return 0.0
    ev = ev_percent / 100.0
    f = (ev / variance) * risk_scale
    cap = 0.5
    if prob < 0.35: cap = 0.025 # 冷門保護
    return min(cap, max(0.0, f)) * 100

def calc_risk_metrics(prob: float, odds: float) -> Tuple[float, float]:
    if prob <= 0 or prob >= 1: return 0.0, 0.0
    win_payoff = odds - 1.0
    lose_payoff = -1.0
    expected_val = prob * win_payoff + (1 - prob) * lose_payoff
    expected_sq = prob * (win_payoff**2) + (1 - prob) * (lose_payoff**2)
    variance = expected_sq - (expected_val**2)
    std_dev = math.sqrt(variance)
    sharpe = expected_val / std_dev if std_dev > 0 else 0
    return variance, sharpe

# =========================
# 2. 全景記憶體系 (Regime Memory)
# =========================
class RegimeMemory:
    def __init__(self):
        self.history_db = {
            "Bore_Draw_Stalemate": { "name": "🛡️ 雙重鐵桶 (悶和局)", "roi": 0.219 }, 
            "Relegation_Dog": { "name": "🐕 保級受讓 (絕境爆發)", "roi": 0.083 },
            "Fallen_Giant": { "name": "📉 豪門崩盤 (名氣大狀況差)", "roi": -0.008 },
            "Fortress_Home": { "name": "🏰 魔鬼主場 (主場過熱)", "roi": -0.008 },
            "Title_MustWin_Home": { "name": "🏆 爭冠必勝盤 (溢價陷阱)", "roi": -0.063 },
            "MarketHype_Fav": { "name": "🔥 大熱倒灶 (過度熱門)", "roi": -0.080 },
            "MidTable_Standard": { "name": "😐 中游例行公事", "roi": 0.000 }
        }

    def analyze_scenario(self, lh: float, la: float, odds: Dict) -> str:
        h_odds = odds.get("home", 2.0)
        prob_h = 1.0 / h_odds
        
        if h_odds < 1.30: return "MarketHype_Fav"
        if (lh + la) < 2.0: return "Bore_Draw_Stalemate"
        if prob_h > 0.6 and h_odds > 1.8: return "Fallen_Giant"
        return "MidTable_Standard"

    def calc_memory_penalty(self, regime_id: str) -> float:
        data = self.history_db.get(regime_id, {"roi": 0})
        roi = data["roi"]
        # V33: 更平滑的懲罰係數
        if roi < -0.05: return 0.8
        if roi > 0.10: return 1.15
        return 1.0

# =========================
# 3. 分析引擎邏輯 (V33.0 Lite Core)
# =========================
class SniperAnalystLogicV33:
    def __init__(self, json_data: Any, max_g: int = 9, nb_alpha: float = 0.12):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha
        self.memory = RegimeMemory()

    def calc_weighted_lambda(self) -> Tuple[float, float]:
        """[V33.0] 近況加權 Lambda 計算"""
        league_base = 1.35 # 聯賽平均基準
        
        # 1. 基礎數據
        def get_base_att_def(team):
            xg = team["offensive_stats"].get("xg_avg", team["offensive_stats"]["goals_scored_avg"])
            xga = team["defensive_stats"].get("xga_avg", team["defensive_stats"]["goals_conceded_avg"])
            # V33: xG 權重提高到 70% (比進球數更準)
            att = 0.3 * team["offensive_stats"]["goals_scored_avg"] + 0.7 * xg
            deff = 0.3 * team["defensive_stats"]["goals_conceded_avg"] + 0.7 * xga
            return att, deff

        h_att, h_def = get_base_att_def(self.h)
        a_att, a_def = get_base_att_def(self.a)
        
        # 2. Time-Decay (近況加權)
        # 解析 recent_form_trend (例如 [1, 0, -1]) -> 權重微調
        def get_form_factor(trend):
            if not trend: return 1.0
            # 越後面的 index 代表越近期的比賽
            # 權重: 最遠(0.1), 中間(0.3), 最近(0.6)
            w = np.linspace(0.1, 0.9, len(trend))
            w /= w.sum()
            score = np.dot(np.array(trend), w) # score 介於 -1 ~ 1
            return 1.0 + (score * 0.1) # 波動範圍 +/- 10%

        h_form = get_form_factor(self.h["context_modifiers"].get("recent_form_trend", []))
        a_form = get_form_factor(self.a["context_modifiers"].get("recent_form_trend", []))
        
        h_att *= h_form
        a_att *= a_form

        # 3. 傷停修正
        if self.h["context_modifiers"].get("missing_key_defender", False): h_def *= 1.25
        if self.a["context_modifiers"].get("missing_key_defender", False): a_def *= 1.20
        
        # 4. 主場優勢
        h_adv = self.h["general_strength"].get("home_advantage_weight", 1.15)
        
        # 5. 最終合成
        lh = (h_att * a_def / league_base) * h_adv
        la = (a_att * h_def / league_base)
        
        return lh, la

    def build_hybrid_matrix(self, lh: float, la: float) -> Tuple[np.ndarray, Dict]:
        """
        [V33.0 核心] 混合矩陣 (Stacking Matrix)
        先生成物理模型矩陣，再根據市場賠率進行權重再平衡 (Rebalancing)。
        """
        # 1. 建立物理模型矩陣 (Base Model Matrix)
        vol_str = self.h.get("style_of_play", {}).get("volatility", "normal")
        vol_adjust = (vol_str == "high")
        M_model = get_base_matrix(lh, la, self.max_g, self.nb_alpha, vol_adjust)
        
        # 2. 獲取市場隱含機率 (去水)
        odds = self.market["1x2_odds"]
        imp_h = implied_prob(odds["home"])
        imp_d = implied_prob(odds["draw"])
        imp_a = implied_prob(odds["away"])
        total_imp = imp_h + imp_d + imp_a
        
        # 歸一化市場機率
        market_probs = {
            "home": imp_h / total_imp,
            "draw": imp_d / total_imp,
            "away": imp_a / total_imp
        }
        
        # 3. 計算模型原始機率
        model_h = float(np.sum(np.tril(M_model, -1)))
        model_d = float(np.sum(np.diag(M_model)))
        model_a = float(np.sum(np.triu(M_model, 1)))
        
        # 4. 混合 (Blending) - 權重設定
        # V33 策略：模型 70% (相信數據) + 市場 30% (尊重莊家)
        w_model = 0.7
        target_h = w_model * model_h + (1 - w_model) * market_probs["home"]
        target_d = w_model * model_d + (1 - w_model) * market_probs["draw"]
        target_a = w_model * model_a + (1 - w_model) * market_probs["away"]
        
        # 5. 矩陣再平衡 (Rebalancing)
        # 將矩陣的三個區域 (下三角、對角線、上三角) 縮放至目標機率
        M_hybrid = M_model.copy()
        
        # 縮放因子
        scale_h = target_h / model_h if model_h > 0 else 0
        scale_d = target_d / model_d if model_d > 0 else 0
        scale_a = target_a / model_a if model_a > 0 else 0
        
        # 應用縮放
        idx_h = np.tril_indices(self.max_g, -1)
        idx_d = np.diag_indices(self.max_g)
        idx_a = np.triu_indices(self.max_g, 1)
        
        M_hybrid[idx_h] *= scale_h
        M_hybrid[idx_d] *= scale_d
        M_hybrid[idx_a] *= scale_a
        
        # 再次歸一化以防誤差
        M_hybrid /= M_hybrid.sum()
        
        return M_hybrid, {
            "model": {"home": model_h, "draw": model_d, "away": model_a},
            "market": market_probs,
            "target": {"home": target_h, "draw": target_d, "away": target_a}
        }

    def ah_ev(self, M: np.ndarray, hcap: float, odds: float) -> float:
        """亞盤四分盤計算 (直接使用混合後的矩陣)"""
        q = int(round(hcap * 4))
        if q % 2 != 0: # Quarter split
            h1 = (q + 1) / 4.0; h2 = (q - 1) / 4.0
            return 0.5 * self.ah_ev(M, h1, odds) + 0.5 * self.ah_ev(M, h2, odds)
        
        G = self.max_g
        idx_diff = np.subtract.outer(np.arange(G), np.arange(G)) 
        r_matrix = idx_diff + hcap
        payoff = np.select(
            [r_matrix > 0.001, np.abs(r_matrix) <= 0.001, r_matrix < -0.001],
            [odds - 1, 0, -1], default=-1
        )
        return np.sum(M * payoff) * 100

    def run_monte_carlo(self, M: np.ndarray, sims: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """從混合矩陣進行聯合抽樣"""
        flat_M = M.flatten()
        flat_M /= flat_M.sum()
        rng = np.random.default_rng(42)
        indices = rng.choice(M.shape[0]**2, size=sims, p=flat_M)
        h_goals = indices // M.shape[0]
        a_goals = indices % M.shape[0]
        results = np.full(sims, "draw", dtype=object)
        results[h_goals > a_goals] = "home"
        results[h_goals < a_goals] = "away"
        return h_goals, a_goals, results.tolist()

# =========================
# 4. Streamlit UI (V33.0 Lite)
# =========================
st.set_page_config(page_title="狙擊手 V33.0 Lite", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 狙擊手 V33.0 Lite (實戰混合版)")
st.markdown("**核心升級**：Hybrid Matrix (市場權重疊加) | Time-Decay (近況加權) | Dynamic Kelly")

# Session State 初始化
if "v33_result" not in st.session_state:
    st.session_state.v33_result = None

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數控制")
    unit_stake = st.number_input("單注本金 ($)", 100, 10000, 100)
    risk_scale = st.slider("風險係數 (Risk Scale)", 0.1, 1.0, 0.4, 0.05)
    nb_alpha = st.slider("Alpha (變異數)", 0.05, 0.25, 0.12, 0.01)
    st.divider()
    st.info("V33.0 自動啟用市場混合模式 (Model 70% + Market 30%)")

# --- 數據輸入區 ---
st.info("請輸入比賽數據 JSON")
default_json = """{
  "meta_info": { "league_name": "英超", "match_date": "2026-03-12" },
  "market_data": {
    "handicaps": [-0.5, 0],
    "goal_lines": [2.5, 3.0],
    "target_odds": 1.95,
    "1x2_odds": { "home": 2.10, "draw": 3.40, "away": 3.40 },
    "opening_odds": { "home": 2.20, "draw": 3.30, "away": 3.20 }
  },
  "home": {
    "name": "主隊 (強勢)",
    "general_strength": { "home_advantage_weight": 1.20 },
    "offensive_stats": { "goals_scored_avg": 1.8, "xg_avg": 1.9 },
    "defensive_stats": { "goals_conceded_avg": 1.1, "xga_avg": 1.0 },
    "style_of_play": { "volatility": "normal" },
    "context_modifiers": { "motivation": "normal", "missing_key_defender": false, "recent_form_trend": [1, 1, 0] }
  },
  "away": {
    "name": "客隊 (低迷)",
    "general_strength": { "home_advantage_weight": 0.9 },
    "offensive_stats": { "goals_scored_avg": 1.0, "xg_avg": 0.9 },
    "defensive_stats": { "goals_conceded_avg": 1.5, "xga_avg": 1.6 },
    "style_of_play": { "volatility": "high" },
    "context_modifiers": { "motivation": "survival", "missing_key_defender": true, "recent_form_trend": [-1, -1, 0] }
  }
}"""
json_input = st.text_area("JSON Input", value=default_json, height=200)

if st.button("🚀 啟動 V33.0 混合運算", type="primary"):
    try:
        input_data = json.loads(json_input)
        engine = SniperAnalystLogicV33(input_data, max_g=9, nb_alpha=nb_alpha)
        
        # 1. 計算加權 Lambda
        lh, la = engine.calc_weighted_lambda()
        
        # 2. 構建混合矩陣 (Hybrid Matrix)
        M_hybrid, probs_info = engine.build_hybrid_matrix(lh, la)
        
        # 3. 記憶體系回溯
        regime_id = engine.memory.analyze_scenario(lh, la, engine.market["1x2_odds"])
        mem_penalty = engine.memory.calc_memory_penalty(regime_id)
        
        st.session_state.v33_result = {
            "engine": engine,
            "M": M_hybrid,
            "lh": lh, "la": la,
            "probs_info": probs_info,
            "regime": regime_id,
            "mem_penalty": mem_penalty
        }
    except Exception as e:
        st.error(f"運算錯誤: {e}")

# --- 結果顯示區 ---
if st.session_state.v33_result:
    res = st.session_state.v33_result
    engine = res["engine"]
    M = res["M"]
    probs = res["probs_info"]
    
    # 頂部儀表板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("主隊預期進球", f"{res['lh']:.2f}")
    c2.metric("客隊預期進球", f"{res['la']:.2f}")
    c3.metric("盤口劇本", res["regime"])
    c4.metric("歷史權重修正", f"x{res['mem_penalty']:.2f}")

    # 混合概率視覺化
    st.subheader("⚖️ Model vs Market 混合權重分析")
    mix_df = pd.DataFrame([
        probs["model"], probs["market"], probs["target"]
    ], index=["純模型 (Physics)", "純市場 (Implied)", "V33 混合 (Hybrid)"])
    st.dataframe(mix_df.style.format("{:.1%}"), use_container_width=True)
    
    if abs(probs["model"]["home"] - probs["market"]["home"]) > 0.15:
        st.warning("⚠️ 警告：模型與市場分歧嚴重，V33 已自動進行權重收斂 (Rebalancing)")
    else:
        st.success("✅ 模型與市場觀點大致相符，信心度高")

    # 投資分析 Tab
    tab_ev, tab_sim = st.tabs(["💰 價值注單 (EV)", "🎲 戰局模擬"])
    
    candidates = []
    
    with tab_ev:
        col_main, col_ah = st.columns([1.2, 1])
        
        with col_main:
            st.markdown("#### 1x2 獨贏 (Hybrid EV)")
            rows_1x2 = []
            for tag, key in [("主勝", "home"), ("和局", "draw"), ("客勝", "away")]:
                prob = probs["target"][key] # 使用混合後的機率
                odds = engine.market["1x2_odds"][key]
                ev = (prob * odds - 1) * 100 * res["mem_penalty"]
                
                var, sharpe = calc_risk_metrics(prob, odds)
                kelly = calc_risk_adj_kelly(ev, var, risk_scale, prob)
                
                rows_1x2.append({"選項": tag, "賠率": odds, "機率": f"{prob:.1%}", "修正EV": f"{ev:+.1f}%", "注碼": f"{kelly:.1f}%"})
                if ev > 1.5:
                    candidates.append({"pick": tag, "ev": ev, "kelly": kelly, "odds": odds, "type": "1x2"})
            
            st.dataframe(pd.DataFrame(rows_1x2), use_container_width=True)

        with col_ah:
            st.markdown("#### 亞盤 & 大小 (Matrix Derived)")
            rows_sub = []
            target_o = engine.market.get("target_odds", 1.95)
            
            # AH
            for hcap in engine.market["handicaps"]:
                ev = engine.ah_ev(M, hcap, target_o) * res["mem_penalty"]
                rows_sub.append({"盤口": f"主 {hcap:+}", "EV": f"{ev:+.1f}%"})
                if ev > 2.0:
                    candidates.append({"pick": f"主 {hcap:+}", "ev": ev, "kelly": calc_risk_adj_kelly(ev, 1.0, risk_scale, 0.5), "odds": target_o, "type": "AH"})
            
            # OU
            G = engine.max_g
            idx_sum = np.add.outer(np.arange(G), np.arange(G))
            for line in engine.market["goal_lines"]:
                prob_over = float(M[idx_sum > line].sum())
                ev_over = (prob_over * target_o - 1) * 100 * res["mem_penalty"]
                rows_sub.append({"盤口": f"大 {line}", "EV": f"{ev_over:+.1f}%"})
                if ev_over > 2.0:
                    candidates.append({"pick": f"大 {line}", "ev": ev_over, "kelly": calc_risk_adj_kelly(ev_over, 1.0, risk_scale, prob_over), "odds": target_o, "type": "OU"})
            
            st.dataframe(pd.DataFrame(rows_sub), use_container_width=True)

        st.divider()
        st.markdown("### 🏆 V33.0 智能推薦")
        if candidates:
            best = sorted(candidates, key=lambda x: x['ev'], reverse=True)[:3]
            for b in best:
                amt = unit_stake * (b['kelly']/100)
                st.markdown(f"""
                <div class="metric-card">
                    <b>{b['type']} {b['pick']}</b> <span style='float:right'>賠率 {b['odds']}</span><br>
                    EV: <span style='color:red'><b>{b['ev']:+.1f}%</b></span> | 建議注碼: <b>{b['kelly']:.1f}%</b> (${amt:.0f})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🚧 暫無高價值注單 (V33 混合模型過濾了低信心機會)")

    with tab_sim:
        st.write("基於混合矩陣 (Hybrid Matrix) 的 5,000 次聯合抽樣")
        h_sim, a_sim, res_sim = engine.run_monte_carlo(M)
        
        sim_df = pd.DataFrame({"Home": h_sim, "Away": a_sim})
        
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown("**進球分佈 (KDE)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(h_sim, bins=range(8), alpha=0.5, label='Home', density=True, color='blue')
            ax.hist(a_sim, bins=range(8), alpha=0.5, label='Away', density=True, color='orange')
            ax.legend()
            st.pyplot(fig)
        
        with c_s2:
            st.markdown("**波膽熱圖 (Top 5)**")
            cs_counts = sim_df.value_counts().head(5).reset_index()
            cs_counts.columns = ["主", "客", "次數"]
            cs_counts["機率"] = (cs_counts["次數"] / 5000).apply(lambda x: f"{x:.1%}")
            st.dataframe(cs_counts)
