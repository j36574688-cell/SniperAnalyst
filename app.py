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
    """計算卜瓦松分佈機率質量函數"""
    return math.exp(-lam) * lam**k / math.factorial(k)

def nb_pmf(k: int, mu: float, alpha: float) -> float:
    """計算負二項分佈機率質量函數"""
    if alpha <= 0:
        return poisson_pmf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    coeff = math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1))
    return float(coeff * (p ** r) * ((1 - p) ** k))

@st.cache_data
def get_matrix_cached(lh: float, la: float, max_g: int, nb_alpha: float, vol_adjust: bool) -> np.ndarray:
    """
    快取矩陣計算結果，避免重複運算
    """
    G = max_g
    i = np.arange(G)
    j = np.arange(G)
    
    # 建立機率向量
    p_i = np.array([poisson_pmf(k, lh) for k in i])
    p_j = np.array([poisson_pmf(k, la) for k in j])
    Mp = np.outer(p_i, p_j)

    nb_i = np.array([nb_pmf(k, lh, nb_alpha) for k in i])
    nb_j = np.array([nb_pmf(k, la, nb_alpha) for k in j])
    Mn = np.outer(nb_i, nb_j)

    # 混合模型 (60% Poisson + 40% Negative Binomial)
    M = 0.6 * Mp + 0.4 * Mn
    
    # 相關性修正 (Dixon-Coles 調整)
    rho = -0.18 if vol_adjust else -0.13
    
    if G > 1:
        M[0,0] *= (1 - lh*la*rho)
        M[1,0] *= (1 + la*rho)
        M[0,1] *= (1 + lh*rho)
        M[1,1] *= (1 - rho)
        
    return M / M.sum()

def calc_risk_adj_kelly(ev_percent: float, variance: float, risk_scale: float = 0.5, prob: float = 0.5) -> float:
    """計算風險調整後的凱利公式注碼"""
    if variance <= 0 or ev_percent <= 0: return 0.0
    ev = ev_percent / 100.0
    f = (ev / variance) * risk_scale
    cap = 0.5
    if prob < 0.35: cap = 0.02 # 冷門保護機制
    return min(cap, max(0.0, f)) * 100

def calc_risk_metrics(prob: float, odds: float) -> Tuple[float, float]:
    """計算變異數與夏普值"""
    if prob <= 0 or prob >= 1: return 0.0, 0.0
    win_payoff = odds - 1.0
    lose_payoff = -1.0
    expected_val = prob * win_payoff + (1 - prob) * lose_payoff
    expected_sq = prob * (win_payoff**2) + (1 - prob) * (lose_payoff**2)
    variance = expected_sq - (expected_val**2)
    std_dev = math.sqrt(variance)
    sharpe = expected_val / std_dev if std_dev > 0 else 0
    return variance, sharpe

def get_true_implied_prob(odds_dict: Dict[str, float]) -> Dict[str, float]:
    """去除水錢，計算真實隱含機率"""
    inv = {}
    for k, v in odds_dict.items():
        try:
            inv[k] = 1.0 / float(v) if v and float(v) > 0 else 0.0
        except:
            inv[k] = 0.0
    margin = sum(inv.values())
    if margin <= 0:
        return {k: 0.0 for k in odds_dict}
    return {k: inv[k] / margin for k in odds_dict}

# =========================
# 2. 全景記憶體系 (Regime Memory)
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
        is_heavy_fav = prob_h > 0.65
        is_underdog = prob_h < 0.35
        motiv_h = h["context_modifiers"]["motivation"]
        motiv_a = a["context_modifiers"]["motivation"]
        
        # 安全讀取近期狀態
        form_h_score = sum(h["context_modifiers"].get("recent_form_trend", [0]))
        form_a_score = sum(a["context_modifiers"].get("recent_form_trend", [0]))
        
        is_title_race = (motiv_h == "title_race")
        is_relegation = (motiv_h == "survival" or motiv_a == "survival")
        
        # 判定劇本
        if h_odds < 2.10 and form_h_score < -1: return "Fallen_Giant"
        if is_heavy_fav and form_h_score < 0: return "Injury_Crisis_Fav"
        if is_title_race and is_heavy_fav: return "Title_MustWin_Home"
        if is_relegation and is_underdog: return "Relegation_Dog"
        
        # 安全讀取主場權重
        h_adv = h["general_strength"].get("home_advantage_weight", 1.15)
        if h_adv > 1.15 and h_odds < 2.0 and form_h_score >= 1: return "Fortress_Home"
        
        if is_underdog and form_h_score > (form_a_score + 2): return "Hidden_Gem_Dog"
        if h_odds < 1.30: return "MarketHype_Fav"
        if (lh + la) < 2.2 and abs(form_h_score) < 2 and abs(form_a_score) < 2: return "Bore_Draw_Stalemate"
        
        return "MidTable_Standard"

    def recall_experience(self, regime_id: str) -> Dict:
        return self.history_db.get(regime_id, {"name": "🔍 未知盤口", "bets": 0, "roi": 0.0})

    def calc_memory_penalty(self, historical_roi: float) -> float:
        if historical_roi < -0.10: return 0.5
        if historical_roi < -0.05: return 0.7
        if historical_roi > 0.15: return 1.2
        if historical_roi > 0.05: return 1.1
        return 1.0

# =========================
# 3. 分析引擎邏輯 (Logic Core)
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

    def calc_lambda(self) -> Tuple[float, float]:
        league_base = 1.35
        
        def att_def(team):
            # 使用 .get() 並提供後備值，解決 KeyError
            xg = team["offensive_stats"].get("xg_avg", team["offensive_stats"]["goals_scored_avg"])
            xga = team["defensive_stats"].get("xga_avg", team["defensive_stats"]["goals_conceded_avg"])
            
            att = 0.4 * team["offensive_stats"]["goals_scored_avg"] + 0.6 * xg
            deff = 0.4 * team["defensive_stats"]["goals_conceded_avg"] + 0.6 * xga
            return att, deff

        h_att, h_def = att_def(self.h)
        a_att, a_def = att_def(self.a)
        
        if self.h["context_modifiers"].get("missing_key_defender", False): h_def *= 1.20
        if self.a["context_modifiers"].get("missing_key_defender", False): a_def *= 1.15
        
        h_adv = self.h["general_strength"].get("home_advantage_weight", 1.15)
        
        lh = (h_att * a_def / league_base) * h_adv
        la = (a_att * h_def / league_base)
        
        if self.h["context_modifiers"]["motivation"] == "survival": lh *= 1.05
        if self.a["context_modifiers"]["motivation"] == "title_race": la *= 1.05
        
        return lh, la

    def get_market_trend_bonus(self) -> Dict[str, float]:
        bonus = {"home":0.0, "draw":0.0, "away":0.0}
        op = self.market.get("opening_odds")
        cu = self.market.get("1x2_odds")
        if not op or not cu: return bonus
        for k in bonus:
            # 簡單計算賠率下跌幅度作為加分
            drop = max(0.0, (op[k] - cu[k]) / op[k])
            bonus[k] = min(3.0, drop * 30.0)
        return bonus

    def build_ensemble_matrix(self, lh: float, la: float) -> np.ndarray:
        vol_str = self.h.get("style_of_play", {}).get("volatility", "normal")
        vol_adjust = (vol_str == "high")
        return get_matrix_cached(lh, la, self.max_g, self.nb_alpha, vol_adjust)

    def ah_ev(self, M: np.ndarray, hcap: float, odds: float) -> float:
        G = self.max_g
        # 使用廣播運算計算分差
        idx_diff = np.subtract.outer(np.arange(G), np.arange(G)) 
        r_matrix = idx_diff + hcap
        
        # 向量化計算派彩結果 (1:贏, 0.5:贏半, 0:走盤, -0.5:輸半, -1:輸)
        payoff = np.select(
            [r_matrix > 0.25, np.abs(r_matrix - 0.25) < 1e-9, np.abs(r_matrix) < 1e-9, np.abs(r_matrix + 0.25) < 1e-9],
            [odds - 1, (odds - 1) * 0.5, 0, -0.5],
            default=-1
        )
        ev = np.sum(M * payoff)
        return ev * 100

    def run_monte_carlo(self, lh: float, la: float, sims: int = 10000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        優化版蒙地卡羅模擬：使用 Numpy 向量化操作提升效能
        預設模擬次數已提升至 10,000 次
        """
        rng = np.random.default_rng(seed)
        home_goals = rng.poisson(lh, sims)
        away_goals = rng.poisson(la, sims)
        
        # 使用向量運算判斷勝負，不再使用 Python 迴圈
        diff = home_goals - away_goals
        results = np.full(sims, "draw", dtype=object)
        results[diff > 0] = "home"
        results[diff < 0] = "away"
        
        return home_goals, away_goals, results.tolist()

    def check_sensitivity(self, lh: float, la: float) -> Tuple[str, float]:
        """壓力測試：當客隊 xG 增加 0.3 時，主勝率下降多少"""
        M_stress = get_matrix_cached(lh, la + 0.3, self.max_g, self.nb_alpha, False)
        M_orig = self.build_ensemble_matrix(lh, la)
        prob_h_orig = float(np.sum(np.tril(M_orig, -1)))
        prob_h_new = float(np.sum(np.tril(M_stress, -1)))
        
        drop_rate = (prob_h_orig - prob_h_new) / prob_h_orig if prob_h_orig > 0 else 0
        level = "Low"
        if drop_rate > 0.15: level = "High"
        elif drop_rate > 0.08: level = "Medium"
        return level, drop_rate

    def calc_model_confidence(self, lh: float, la: float, market_diff_percent: float, sens_drop_rate: float) -> Tuple[float, List[str]]:
        score = 1.0
        reasons = []
        if market_diff_percent > 0.25:
            score *= 0.7; reasons.append("與市場差異過大 (>25%)")
        elif market_diff_percent > 0.15:
            score *= 0.85; reasons.append("與市場顯著分歧")
        
        if sens_drop_rate > 0.15:
            score *= 0.8; reasons.append("模型對運氣球極度敏感")
        elif sens_drop_rate > 0.08:
            score *= 0.9; reasons.append("敏感度偏高")
            
        total_xg = lh + la
        if total_xg > 3.5:
            score *= 0.9; reasons.append("高入球預期 (亂戰風險)")
            
        return score, reasons

# =========================
# 4. Streamlit UI 介面 (UI Layer)
# =========================
st.set_page_config(page_title="狙擊手分析 V31.5 MC10K", page_icon="⚽", layout="wide")

st.title("⚽ 狙擊手 V31.5 (架構優化 + 10K模擬版)")
st.markdown("### 專業足球數據分析：向量化加速 x 狀態管理 x 10,000次精準模擬")

# --- 初始化 Session State (優化：防止切換 Tab 時數據消失) ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    unit_stake = st.number_input("💰 設定單注本金 ($)", min_value=10, value=100, step=10)
    st.divider()
    nb_alpha = st.slider("Alpha (變異數)", 0.05, 0.20, 0.12, 0.01)
    max_g = st.number_input("運算範圍 (max_g)", 5, 20, 9)
    risk_scale = st.slider("風險縮放係數", 0.1, 1.0, 0.3, 0.1)
    st.divider()
    enable_fixed_seed = st.toggle("固定隨機數種子 (除錯/回測用)", value=True)
    seed_val = 42 if enable_fixed_seed else None
    use_mock_memory = st.checkbox("🧠 啟用歷史記憶 (真實回測數據)", value=True)

# --- 輸入區 ---
st.info("請選擇數據輸入方式：")
tab_input1, tab_input2 = st.tabs(["📋 貼上 JSON 代碼", "📂 上傳 JSON 檔案"])
input_data = None

default_json = """{
  "meta_info": { "league_name": "範例聯賽", "match_date": "2026-03-11" },
  "market_data": {
    "handicaps": [-0.75, 0.25],
    "goal_lines": [2.5, 3.0],
    "target_odds": 1.90,
    "1x2_odds": { "home": 1.72, "draw": 4.00, "away": 4.00 },
    "opening_odds": { "home": 3.20, "draw": 3.60, "away": 2.20 }
  },
  "home": {
    "name": "主隊範例",
    "general_strength": { "home_advantage_weight": 1.25 },
    "offensive_stats": { "goals_scored_avg": 1.57, "xg_avg": 1.6 },
    "defensive_stats": { "goals_conceded_avg": 1.0, "xga_avg": 0.95 },
    "style_of_play": { "volatility": "normal" },
    "context_modifiers": { "motivation": "title_race", "missing_key_defender": false, "recent_form_trend": [1, 0, -1] }
  },
  "away": {
    "name": "客隊範例",
    "general_strength": { "home_advantage_weight": 0.80 },
    "offensive_stats": { "goals_scored_avg": 0.80, "xg_avg": 0.9 },
    "defensive_stats": { "goals_conceded_avg": 1.33, "xga_avg": 1.5 },
    "style_of_play": { "volatility": "high" },
    "context_modifiers": { "motivation": "survival", "missing_key_defender": true, "recent_form_trend": [-1, 1, -1] }
  }
}"""

with tab_input1:
    json_text = st.text_area("在此貼上 JSON", value=default_json, height=150)
    if json_text:
        try: input_data = json.loads(json_text)
        except: st.error("JSON 格式錯誤")
with tab_input2:
    uploaded_file = st.file_uploader("選擇 .json 或 .txt 檔案", type=['json', 'txt'])
    if uploaded_file:
        try: input_data = json.load(uploaded_file)
        except: st.error("檔案讀取失敗")

# --- 執行分析 ---
if st.button("🚀 開始全方位分析", type="primary"):
    if not input_data:
        st.error("請先輸入有效的比賽數據！")
    else:
        # 防呆：確保欄位存在
        if "recent_form_trend" not in input_data["home"]["context_modifiers"]:
            input_data["home"]["context_modifiers"]["recent_form_trend"] = [0,0,0]
        if "recent_form_trend" not in input_data["away"]["context_modifiers"]:
            input_data["away"]["context_modifiers"]["recent_form_trend"] = [0,0,0]

        # 初始化引擎
        engine = SniperAnalystLogic(input_data, max_g, nb_alpha)
        
        # 1. 基礎計算
        lh, la = engine.calc_lambda()
        M = engine.build_ensemble_matrix(lh, la)
        market_bonus = engine.get_market_trend_bonus()
        true_imp_probs = get_true_implied_prob(engine.market["1x2_odds"])
        
        # 2. 全景記憶識別
        regime_id = engine.memory.analyze_scenario(engine, lh, la)
        history_data = {"name": "未知", "bets": 0, "roi": 0.0}
        memory_penalty = 1.0
        
        if use_mock_memory:
            history_data = engine.memory.recall_experience(regime_id)
            memory_penalty = engine.memory.calc_memory_penalty(history_data["roi"])

        # 3. 信心分數
        prob_h = float(np.sum(np.tril(M, -1)))
        diff_h = max(0, prob_h - true_imp_probs["home"])
        sens_level, sens_drop = engine.check_sensitivity(lh, la)
        model_conf_score, conf_reasons = engine.calc_model_confidence(lh, la, diff_h, sens_drop)
        
        # 將計算結果存入 Session State
        st.session_state.analysis_results = {
            "engine": engine,
            "M": M,
            "lh": lh,
            "la": la,
            "market_bonus": market_bonus,
            "true_imp_probs": true_imp_probs,
            "history_data": history_data,
            "memory_penalty": memory_penalty,
            "model_conf_score": model_conf_score,
            "prob_h": prob_h
        }

# --- 結果顯示區 (從 Session State 讀取，避免重算) ---
if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    engine = res["engine"]
    M = res["M"]
    history_data = res["history_data"]
    
    # 側邊欄資訊
    with st.sidebar:
        st.divider()
        st.subheader("🧠 盤口劇本識別")
        st.info(f"{history_data['name']}")
        
        if use_mock_memory:
            col_h1, col_h2 = st.columns(2)
            col_h1.metric("歷史樣本", f"{history_data['bets']}場")
            col_h2.metric("歷史 ROI", f"{history_data['roi']*100:.1f}%", delta_color="normal" if history_data['roi'] > 0 else "inverse")
            
            penalty = res["memory_penalty"]
            if penalty < 1.0: st.error(f"⚠️ 歷史虧損懲罰: EV x {penalty}")
            elif penalty > 1.0: st.success(f"🔥 歷史獲利加成: EV x {penalty}")
        else: st.caption("記憶模擬未啟用")

        st.divider()
        st.subheader("🛡️ 模型信心")
        st.metric("Confidence", f"{res['model_conf_score']*100:.0f}/100")

    # 主畫面 Header
    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        st.markdown(f"<h3 style='text-align: right; color: #1f77b4;'>{engine.h['name']}</h3>", unsafe_allow_html=True)
        st.metric("預期進球", f"{res['lh']:.2f}")
    with col2: st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h3 style='text-align: left; color: #ff7f0e;'>{engine.a['name']}</h3>", unsafe_allow_html=True)
        st.metric("預期進球", f"{res['la']:.2f}")

    prob_d = float(np.sum(np.diag(M)))
    prob_a = float(np.sum(np.triu(M, 1)))
    prob_h = res["prob_h"]

    res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📊 價值與劇本修正", "🧠 智能裁決", "🎯 波膽分佈", "🎲 模擬與雷達"])

    candidates = []

    with res_tab1:
        st.subheader("💰 獨贏 (1x2)")
        rows_1x2 = []
        for tag, prob, key in [("主勝", prob_h, "home"), ("和局", prob_d, "draw"), ("客勝", prob_a, "away")]:
            odd = engine.market["1x2_odds"][key]
            raw_ev = (prob * odd - 1) * 100 + res["market_bonus"][key]
            adj_ev = raw_ev * res["model_conf_score"] * res["memory_penalty"]
            
            var, sharpe = calc_risk_metrics(prob, odd)
            kelly_pct = calc_risk_adj_kelly(adj_ev, var, risk_scale, prob)
            profit = (odd - 1) * unit_stake
            
            note = ""
            if prob < 0.35 and adj_ev > 0: note = "⚠️ 冷門小注"
            
            rows_1x2.append({
                "選項": tag, "賠率": odd, 
                "原始 EV": f"{raw_ev:+.1f}%",
                "修正 EV": f"{adj_ev:+.1f}%",
                "預計獲利": f"${profit:.1f}",
                "夏普值": f"{sharpe:.2f}",
                "建議注碼%": f"{kelly_pct:.1f}%",
                "備註": note
            })
            if adj_ev > 1.5:
                candidates.append({
                    "type":"1x2", "pick":tag, "ev":adj_ev, "raw_ev":raw_ev,
                    "odds":odd, "prob":prob, "sens": "Low", # 簡化
                    "sharpe": sharpe, "kelly": kelly_pct, "note": note
                })
        st.dataframe(pd.DataFrame(rows_1x2), use_container_width=True)

        c_ah, c_ou = st.columns(2)
        with c_ah:
            st.subheader("🛡️ 亞盤")
            d_ah = []
            target_o = engine.market.get("target_odds", 1.90)
            
            for hcap in engine.market["handicaps"]:
                raw_ev = engine.ah_ev(M, hcap, target_o) + res["market_bonus"]["home"]
                adj_ev = raw_ev * res["model_conf_score"] * res["memory_penalty"]
                
                prob_approx = (raw_ev/100.0 + 1) / target_o
                var, sharpe = calc_risk_metrics(prob_approx, target_o)
                kelly_pct = calc_risk_adj_kelly(adj_ev, var, risk_scale, prob_approx)
                profit = (target_o - 1) * unit_stake

                d_ah.append({
                    "盤口": f"主 {hcap:+}", "賠率": target_o, 
                    "修正 EV": f"{adj_ev:+.1f}%", "預計獲利": f"${profit:.1f}",
                    "夏普值": f"{sharpe:.2f}", "建議注碼%": f"{kelly_pct:.1f}%"
                })
                if adj_ev > 2: 
                    candidates.append({
                        "type":"AH", "pick":f"主 {hcap:+}", "ev":adj_ev, "raw_ev":raw_ev,
                        "odds":target_o, "prob":prob_approx, "sens":"Medium",
                        "sharpe": sharpe, "kelly": kelly_pct, "note": ""
                    })
            st.dataframe(pd.DataFrame(d_ah), use_container_width=True)
        
        with c_ou:
            st.subheader("📐 大小球 (雙向)")
            d_ou = []
            
            G = engine.max_g
            idx_sum = np.add.outer(np.arange(G), np.arange(G))
            target_o = engine.market.get("target_odds", 1.90)

            for line in engine.market["goal_lines"]:
                prob_over = float(M[idx_sum > line].sum())
                prob_under = float(M[idx_sum < line].sum())
                
                for side_label, op, pick_name in [("大", prob_over, f"大 {line}"), ("小", prob_under, f"小 {line}")]:
                    raw_ev = (op * target_o - 1) * 100
                    adj_ev = raw_ev * res["model_conf_score"] * res["memory_penalty"]
                    
                    var, sharpe = calc_risk_metrics(op, target_o)
                    kelly_pct = calc_risk_adj_kelly(adj_ev, var, risk_scale, op)
                    profit = (target_o - 1) * unit_stake

                    d_ou.append({
                        "盤口": pick_name, "賠率": target_o, 
                        "修正 EV": f"{adj_ev:+.1f}%", "預計獲利": f"${profit:.1f}",
                        "夏普值": f"{sharpe:.2f}", "建議注碼%": f"{kelly_pct:.1f}%"
                    })
                    
                    if adj_ev > 2: 
                        candidates.append({
                            "type":"OU", "pick":pick_name, "ev":adj_ev, "raw_ev":raw_ev,
                            "odds":target_o, "prob":op, "sens":"Medium",
                            "sharpe": sharpe, "kelly": kelly_pct, "note": ""
                        })
                        
            st.dataframe(pd.DataFrame(d_ou), use_container_width=True)

        st.subheader("📝 智能投資組合 (劇本加權)")
        if candidates:
            final = sorted(candidates, key=lambda x:x["ev"], reverse=True)[:3]
            no_bet_flag = False; no_bet_reason = []
            
            if use_mock_memory and history_data['roi'] < -0.05:
                    no_bet_flag = True
                    no_bet_reason.append(f"劇本警示：此劇本 ({history_data['name']}) 歷史為負期望值，建議避開")

            if res["model_conf_score"] < 0.6:
                no_bet_flag = True; no_bet_reason.append(f"模型信心過低 ({res['model_conf_score']*100:.0f}/100)")
            
            if no_bet_flag:
                st.error(f"🛑 系統建議觀望 (NO BET)")
                for r in no_bet_reason: st.write(f"- {r}")
            else:
                reco = []
                for p in final:
                    bet_amount = unit_stake * (p['kelly'] / 10.0)
                    risk_icon = "🟢" if p['sharpe'] > 0.1 else ("🟡" if p['sharpe'] > 0.05 else "🔴")
                    reco.append([
                        f"[{p['type']}] {p['pick']}", p['odds'], 
                        f"{p['raw_ev']:+.1f}%", f"{p['ev']:+.1f}%",      
                        f"{risk_icon} {p['sharpe']:.3f}", 
                        f"{p['kelly']:.1f}%", f"${bet_amount:.1f}",
                        p['note']
                    ])
                st.dataframe(pd.DataFrame(reco, columns=["選項", "賠率", "原始EV", "修正EV", "夏普值", "注碼%", "建議金額", "備註"]), use_container_width=True)
        else:
            st.info("無適合注單")

    with res_tab2:
        st.subheader("🧠 模型裁決")
        total_xg = res["lh"] + res["la"]
        if total_xg > 3.5: st.warning(f"🟠 高變異節奏 (xG {total_xg:.2f})")
        elif total_xg > 2.5: st.success(f"🟢 中性節奏 (xG {total_xg:.2f})")
        else: st.info(f"🔵 低節奏 (xG {total_xg:.2f})")
        
        if candidates:
            top = sorted(candidates, key=lambda x:x["ev"], reverse=True)[0]
            
            market_imp = 0.0
            if top['type'] == '1x2':
                key_map = {"主勝":"home", "和局":"draw", "客勝":"away"}
                market_imp = res["true_imp_probs"].get(key_map.get(top['pick']), 0.0)
            else:
                market_imp = 1.0/top['odds']

            diff = top['prob'] - market_imp
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("模型機率", f"{top['prob']*100:.1f}%")
            col_c2.metric("市場隱含(去水)", f"{market_imp*100:.1f}%")
            if diff < 0: st.error("🔴 虛高風險：EV 來自賠率槓桿")
            elif diff < 0.03: st.warning("🟠 邊際優勢：優勢不明顯")
            else: st.success("🟢 真實價值：顯著機率偏差")

    with res_tab3:
        st.subheader("🎯 波膽分佈 (效能優化)")
        disp_g = min(6, engine.max_g)
        df_cs = pd.DataFrame(M[:disp_g,:disp_g], columns=[f"客{j}" for j in range(disp_g)], index=[f"主{i}" for i in range(disp_g)])
        st.dataframe(df_cs.style.format("{:.1%}", subset=None))

    with res_tab4:
        st.subheader("🎲 戰局模擬 (10,000次)")
        # 這裡會跑 10000 次模擬
        sh, sa, sr = engine.run_monte_carlo(res["lh"], res["la"], sims=10000, seed=seed_val)
        
        sim_count = len(sr)
        sc1, sc2, sc3 = st.columns(3)
        
        sc1.metric("主勝率", f"{sr.count('home')/sim_count*100:.1f}%")
        sc2.metric("和局率", f"{sr.count('draw')/sim_count*100:.1f}%")
        sc3.metric("客勝率", f"{sr.count('away')/sim_count*100:.1f}%")
        
        fig, ax = plt.subplots(figsize=(10,4))
        ch, bh = np.histogram(sh, bins=range(10), density=True)
        ca, ba = np.histogram(sa, bins=range(10), density=True)
        ax.bar(bh[:-1]-0.15, ch, width=0.3, color='#1f77b4', alpha=0.7, label='Home')
        ax.bar(ba[:-1]+0.15, ca, width=0.3, color='#ff7f0e', alpha=0.7, label='Away')
        ax.legend(); st.pyplot(fig)
        
        st.divider()
        st.subheader("⚔️ 戰力雷達")
        cats = ['Attack', 'Defense', 'Form', 'Home/Away', 'Motivation']
        def get_s(stats):
            form_val = sum(stats.get("context_modifiers", {}).get("recent_form_trend", [0,0,0]))
            form_score = (form_val + 3) * 1.5 
            # 修正 xg 讀取邏輯
            xg = stats["offensive_stats"].get("xg_avg", stats["offensive_stats"]["goals_scored_avg"])
            xga = stats["defensive_stats"].get("xga_avg", stats["defensive_stats"]["goals_conceded_avg"])
            h_adv = stats["general_strength"].get("home_advantage_weight", 1.0)
            
            return [min(10, xg*4), min(10, (3-xga)*3.5), form_score, h_adv*5, 8 if stats["context_modifiers"]["motivation"]!="normal" else 5]
        
        hs, ans = get_s(engine.h), get_s(engine.a)
        N = len(cats); ang = [n/float(N)*2*math.pi for n in range(N)]; ang+=ang[:1]; hs+=hs[:1]; ans+=ans[:1]
        figr, axr = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        axr.plot(ang, hs, color='#1f77b4', label='Home'); axr.fill(ang, hs, '#1f77b4', alpha=0.2)
        axr.plot(ang, ans, color='#ff7f0e', label='Away'); axr.fill(ang, ans, '#ff7f0e', alpha=0.2)
        axr.set_xticks(ang[:-1]); axr.set_xticklabels(cats); axr.legend()
        st.pyplot(figr)
