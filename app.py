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

# [V38] 嘗試導入 Numba 進行 JIT 加速
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(fastmath=False, parallel=False):
        def decorator(func): return func
        return decorator
    def prange(n): return range(n)

# =========================
# 1. 核心數學工具 (V38.5 Kernel)
# =========================
EPS = 1e-15

@njit(fastmath=True)
def fast_log_factorial(n):
    if n < 0: return 0.0
    if n <= 20:
        res = 0.0
        for i in range(1, n + 1): res += math.log(i)
        return res
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

@njit(fastmath=True)
def poisson_logpmf_fast(k, lam):
    if lam <= 0: return 0.0 if k == 0 else -1e10
    return -lam + k * math.log(lam) - fast_log_factorial(k)

@njit(fastmath=True)
def biv_poisson_logpmf_fast(x, y, lam1, lam2, lam3):
    if lam3 <= 1e-9: return poisson_logpmf_fast(x, lam1) + poisson_logpmf_fast(y, lam2)
    base = -(lam1 + lam2 + lam3)
    max_val = -1e20
    terms = np.zeros(min(x, y) + 1)
    for k in range(min(x, y) + 1):
        t = base
        if x-k>0: t += (x-k)*math.log(lam1) - fast_log_factorial(x-k)
        if y-k>0: t += (y-k)*math.log(lam2) - fast_log_factorial(y-k)
        if k>0: t += k*math.log(lam3) - fast_log_factorial(k)
        terms[k] = t
        if t > max_val: max_val = t
    sum_exp = 0.0
    for i in range(len(terms)): sum_exp += math.exp(terms[i] - max_val)
    return max_val + math.log(sum_exp)

# [V38.3] 向量化 NLL 計算
@njit(fastmath=True, parallel=True)
def compute_batch_nll(lh_arr, la_arr, h_arr, a_arr, lam3, rho, home_adv):
    nll = 0.0
    n = len(lh_arr)
    for i in prange(n):
        lh = lh_arr[i] * home_adv
        la = la_arr[i]
        h = h_arr[i]
        a = a_arr[i]
        l1 = max(0.01, lh - lam3)
        l2 = max(0.01, la - lam3)
        lp = biv_poisson_logpmf_fast(h, a, l1, l2, lam3)
        prob = math.exp(lp)
        if h==0 and a==0: prob *= (1 - lh*la*rho)
        elif h==0 and a==1: prob *= (1 + lh*rho)
        elif h==1 and a==0: prob *= (1 + la*rho)
        elif h==1 and a==1: prob *= (1 - rho)
        if prob > 1e-9: nll -= math.log(prob)
        else: nll -= math.log(1e-9)
    return nll

def get_true_implied_prob(odds_dict):
    inv = {k: 1.0/v if v>0 else 0.0 for k,v in odds_dict.items()}
    s = sum(inv.values())
    return {k: inv[k]/s if s>0 else 0.0 for k in odds_dict}

def calc_risk_adj_kelly(ev_percent, variance, risk_scale=0.5, prob=0.5):
    if variance<=0 or ev_percent<=0: return 0.0
    ev = ev_percent/100.0
    f = (ev / variance) * risk_scale
    cap = 0.5 if prob>=0.35 else 0.025
    return min(cap, max(0.0, f)) * 100

def calc_risk_metrics(prob, odds):
    if prob<=0 or prob>=1: return 0.0, 0.0
    win_p, lose_p = odds-1.0, -1.0
    ev = prob*win_p + (1-prob)*lose_p
    var = prob*(win_p**2) + (1-prob)*(lose_p**2) - (ev**2)
    sharpe = ev/math.sqrt(var) if var>0 else 0
    return var, sharpe

@st.cache_data
def get_matrix_cached(lh, la, max_g, nb_alpha):
    # Fallback for sensitivity check
    G = max_g
    M = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            # Simple Poisson for stress test
            p = math.exp(biv_poisson_logpmf_fast(i, j, lh, la, 0.0))
            M[i, j] = p
    return M / M.sum()

# =========================
# 2. 全景記憶體系
# =========================
class RegimeMemory:
    def __init__(self, db_path="regime_db.json"):
        self.db_path = db_path
        self.default_db = {
            "Bore_Draw_Stalemate": { "name": "🛡️ 雙重鐵桶", "roi": 0.219, "bets": 2150 }, 
            "Relegation_Dog": { "name": "🐕 保級受讓", "roi": 0.083, "bets": 1840 },
            "Fallen_Giant": { "name": "📉 豪門崩盤", "roi": -0.008, "bets": 920 },
            "Fortress_Home": { "name": "🏰 魔鬼主場", "roi": -0.008, "bets": 3100 },
            "Title_MustWin_Home": { "name": "🏆 爭冠必勝盤", "roi": -0.063, "bets": 2450 },
            "MarketHype_Fav": { "name": "🔥 大熱倒灶", "roi": -0.080, "bets": 1560 },
            "MidTable_Standard": { "name": "😐 中游例行", "roi": 0.000, "bets": 5000 }
        }
        self.history_db = self.load_db()

    def load_db(self) -> Dict:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f: return json.load(f)
            except: return self.default_db
        return self.default_db

    def analyze_scenario(self, lh, la, odds) -> str:
        h = odds.get("home", 2.0)
        if h < 1.30: return "MarketHype_Fav"
        if (lh+la) < 2.2: return "Bore_Draw_Stalemate"
        if h < 2.0: return "Fortress_Home"
        return "MidTable_Standard"

    def recall_experience(self, rid: str) -> Dict:
        return self.history_db.get(rid, {"name": "未知", "roi": 0.0, "bets": 0})

    def calc_memory_penalty(self, roi: float) -> float:
        if roi < -0.05: return 0.7
        if roi > 0.05: return 1.1
        return 1.0

# =========================
# 3. 分析引擎邏輯 (V38.5 Restored Full Logic)
# =========================
class SniperAnalystLogic:
    def __init__(self, json_data, max_g=9, nb_alpha=0.12, lam3=0.0, rho=-0.13, home_adv=1.15):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha
        self.lam3, self.rho, self.home_adv = lam3, rho, home_adv
        self.memory = RegimeMemory()

    def calc_lambda(self):
        # [Restored Logic]
        def att_def_w(team):
            xg, xga = team["offensive_stats"].get("xg_avg", 1.0), team["defensive_stats"].get("xga_avg", 1.0)
            trend = team["context_modifiers"].get("recent_form_trend", [0, 0, 0])
            w = np.array([0.1, 0.3, 0.6])
            form_factor = 1.0 + (np.dot(trend[-len(w):], w[-len(trend):]) * 0.1)
            return (0.3 * team["offensive_stats"]["goals_scored_avg"] + 0.7 * xg) * form_factor, \
                   (0.3 * team["defensive_stats"]["goals_conceded_avg"] + 0.7 * xga)

        lh_att, lh_def = att_def_w(self.h)
        la_att, la_def = att_def_w(self.a)
        
        strength_gap = (lh_att - la_att)
        crush_factor = 1.05 if strength_gap > 0.5 else 1.0
        
        # Apply Home Adv
        lh = (lh_att * la_def / 1.35) * self.home_adv * crush_factor
        la = (la_att * lh_def / 1.35)
        
        if self.h["context_modifiers"].get("missing_key_defender"): lh *= 0.9 # logic changed? assume def affects other team score
        # Previous logic: if home missing defender, AWAY score increases (la increases)
        if self.h["context_modifiers"].get("missing_key_defender"): la *= 1.25
        if self.a["context_modifiers"].get("missing_key_defender"): lh *= 1.20
        
        return lh, la, True

    def build_matrix_v38(self, lh, la, use_biv=True, use_dc=True):
        G = self.max_g
        M = np.zeros((G, G))
        l3 = max(self.lam3, 0.001) if use_biv else 0.0
        l1, l2 = max(0.01, lh-l3), max(0.01, la-l3)
        
        for i in range(G):
            for j in range(G):
                M[i,j] = math.exp(biv_poisson_logpmf_fast(i, j, l1, l2, l3))
        
        if use_dc:
            rho = self.rho
            def tau(x, y):
                if x==0 and y==0: return 1 - lh*la*rho
                elif x==0 and y==1: return 1 + lh*rho
                elif x==1 and y==0: return 1 + la*rho
                elif x==1 and y==1: return 1 - rho
                return 1.0
            for i in range(2):
                for j in range(2): M[i,j] *= tau(i,j)
            
        M /= M.sum()
        
        imp = get_true_implied_prob(self.market["1x2_odds"])
        ph, pd, pa = float(np.sum(np.tril(M,-1))), float(np.sum(np.diag(M))), float(np.sum(np.triu(M,1)))
        
        w = 0.7 if abs(ph - imp["home"]) < 0.2 else 0.5
        th = w*ph + (1-w)*imp["home"]
        td = w*pd + (1-w)*imp["draw"]
        ta = w*pa + (1-w)*imp["away"]
        
        M_hybrid = M.copy()
        M_hybrid[np.tril_indices(G,-1)] *= (th/ph if ph>0 else 1)
        M_hybrid[np.diag_indices(G)] *= (td/pd if pd>0 else 1)
        M_hybrid[np.triu_indices(G,1)] *= (ta/pa if pa>0 else 1)
        M_hybrid /= M_hybrid.sum()
        
        return M_hybrid, {"model": {"home": ph, "draw": pd, "away": pa}, "market": imp, "hybrid": {"home": th, "draw": td, "away": ta}}

    def get_market_trend_bonus(self):
        # [Restored]
        bonus = {"home":0.0, "draw":0.0, "away":0.0}
        op, cu = self.market.get("opening_odds"), self.market.get("1x2_odds")
        if not op or not cu: return bonus
        for k in bonus:
            drop = max(0.0, (op[k] - cu[k]) / op[k])
            bonus[k] = min(3.0, drop * 30.0)
        return bonus

    def ah_ev(self, M, hcap, odds):
        # [Restored Recursive Logic for 0.25/0.75]
        q = int(round(hcap * 4))
        if q % 2 != 0: return 0.5 * self.ah_ev(M, (q+1)/4.0, odds) + 0.5 * self.ah_ev(M, (q-1)/4.0, odds)
        
        idx_diff = np.subtract.outer(np.arange(self.max_g), np.arange(self.max_g)) 
        payoff = np.select([idx_diff + hcap > 0.001, np.abs(idx_diff + hcap) <= 0.001], [odds-1, 0], default=-1)
        return np.sum(M * payoff) * 100

    def check_sensitivity(self, lh, la):
        # [Restored]
        M_stress = get_matrix_cached(lh, la + 0.3, self.max_g, self.nb_alpha)
        p_orig = float(np.sum(np.tril(get_matrix_cached(lh, la, self.max_g, self.nb_alpha), -1)))
        p_new = float(np.sum(np.tril(M_stress, -1)))
        drop = (p_orig - p_new) / p_orig if p_orig > 0 else 0
        return ("High" if drop > 0.15 else "Medium"), drop

    def calc_model_confidence(self, lh, la, diff, sens):
        # [Restored]
        score, reasons = 1.0, []
        if diff > 0.25: score *= 0.7; reasons.append(f"與市場差異過大 ({diff:.1%})")
        if sens > 0.15: score *= 0.8; reasons.append("模型對運氣球敏感")
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

    def run_monte_carlo_vectorized(self, M, sims=500000):
        rng = np.random.default_rng()
        flat = M.flatten(); flat /= flat.sum()
        cdf = np.cumsum(flat)
        idx = np.searchsorted(cdf, rng.random(sims))
        hg, ag = idx // M.shape[0], idx % M.shape[0]
        return np.sum(hg>ag)/sims, np.sum(hg==ag)/sims, np.sum(hg<ag)/sims, hg, ag

    def run_ce_importance_sampling(self, M, line, n_sims=20000):
        G = M.shape[0]
        # Calculate means
        i_idx, j_idx = np.indices((G,G))
        mu_h = np.sum(M * i_idx)
        mu_a = np.sum(M * j_idx)
        
        # Biased params
        v_h, v_a = mu_h * 1.5, mu_a * 1.5
        rng = np.random.default_rng()
        sh = rng.poisson(v_h, n_sims)
        sa = rng.poisson(v_a, n_sims)
        
        # Likelihood Ratio
        log_w = (sh*(np.log(mu_h)-np.log(v_h)) - (mu_h-v_h)) + \
                (sa*(np.log(mu_a)-np.log(v_a)) - (mu_a-v_a))
        w = np.exp(log_w)
        
        est = np.sum(w * ((sh+sa)>line)) / n_sims
        return {"est": float(est)}

# =========================
# 4. 資料處理工具 (V38.4 Auto-Adapter)
# =========================
def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        'HomeTeam': 'home', 'Home': 'home', 'HT': 'home',
        'AwayTeam': 'away', 'Away': 'away', 'AT': 'away',
        'FTHG': 'home_goals', 'HG': 'home_goals', 'HomeGoals': 'home_goals',
        'FTAG': 'away_goals', 'AG': 'away_goals', 'AwayGoals': 'away_goals',
        'Div': 'div', 'Date': 'date'
    }
    df.columns = [c.strip() for c in df.columns]
    new_cols = {}
    for col in df.columns:
        for k, v in col_map.items():
            if col.lower() == k.lower():
                new_cols[col] = v; break
    df = df.rename(columns=new_cols)
    
    required = ['home', 'away', 'home_goals', 'away_goals']
    if any(c not in df.columns for c in required):
        st.error(f"❌ 缺少關鍵欄位: {required}")
        return pd.DataFrame()

    if 'lh_pred' not in df.columns or 'la_pred' not in df.columns:
        st.info("ℹ️ 自動生成預期進球 (Based on League Avg)...")
        avg_h, avg_a = df['home_goals'].mean(), df['away_goals'].mean()
        df['lh_pred'] = avg_h; df['la_pred'] = avg_a
        # Try simple rolling mean
        try:
            h_roll = df.groupby('home')['home_goals'].transform(lambda x: x.shift().expanding().mean())
            a_roll = df.groupby('away')['away_goals'].transform(lambda x: x.shift().expanding().mean())
            df['lh_pred'] = h_roll.fillna(avg_h)
            df['la_pred'] = a_roll.fillna(avg_a)
        except: pass
    return df

def fit_params_mle(df):
    if df.empty: return {"success": False}
    try:
        lh_arr = df['lh_pred'].values.astype(np.float64)
        la_arr = df['la_pred'].values.astype(np.float64)
        h_arr = df['home_goals'].values.astype(np.int32)
        a_arr = df['away_goals'].values.astype(np.int32)
    except: return {"success": False}

    def nll_func(params):
        lam3, rho, ha = params
        if not (0<=lam3<=0.5 and -0.3<=rho<=0.3 and 0.8<=ha<=1.6): return 1e9
        return compute_batch_nll(lh_arr, la_arr, h_arr, a_arr, lam3, rho, ha)

    res = minimize(nll_func, [0.1, -0.1, 1.15], method='Nelder-Mead', tol=1e-3)
    return {"lam3": res.x[0], "rho": res.x[1], "home_adv": res.x[2], "success": res.success}

def run_kalman_tracking(df):
    class SimpleKalmanFilter:
        def __init__(self, r=1.0): self.x=r; self.P=1.0; self.Q=0.05; self.R=1.0
        def predict(self): self.P+=self.Q; return self.x
        def update(self, z):
            K = self.P/(self.P+self.R)
            self.x += K*(z-self.x)
            self.P *= (1-K)
            return self.x
    
    if df.empty: return pd.DataFrame(), {}
    teams = set(df['home']).union(set(df['away']))
    ratings = {t: SimpleKalmanFilter() for t in teams}
    hist = []
    for _, r in df.iterrows():
        h, a = r['home'], r['away']
        rh, ra = ratings[h].predict(), ratings[a].predict()
        n_h, n_a = ratings[h].update(r['home_goals']), ratings[a].update(r['away_goals'])
        hist.append({'home': h, 'away': a, 'h_rating': n_h, 'a_rating': n_a})
    return pd.DataFrame(hist), ratings

# =========================
# 5. UI (V38.5 Restoration)
# =========================
st.set_page_config(page_title="Sniper V38.5", page_icon="🧿", layout="wide")
st.markdown("<style>.metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; } .stProgress > div > div > div > div { background-color: #4CAF50; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧿 Sniper V38.5")
    st.caption("Restored Edition")
    if HAS_NUMBA: st.success("⚡ Numba 加速：ON")
    else: st.warning("⚠️ Numba 加速：OFF")
    
    app_mode = st.radio("功能模式：", ["🎯 單場深度預測", "🛡️ 風險對沖實驗室", "🔧 參數校正實驗室", "📈 聯賽歷史回測", "📚 劇本查詢"])
    st.divider()
    with st.expander("🛠️ 進階參數", expanded=False):
        unit_stake = st.number_input("單注 ($)", 10, 10000, 100)
        nb_alpha = st.slider("Alpha", 0.05, 0.25, 0.12)
        use_biv = st.toggle("Biv Poisson", True)
        use_dc = st.toggle("Dixon-Coles", True)
        st.markdown("---")
        lam3_in = st.number_input("Lambda 3", 0.0, 0.5, 0.15, step=0.01)
        rho_in = st.number_input("Rho", -0.3, 0.3, -0.13, step=0.01)
        ha_in = st.number_input("Home Adv", 0.8, 1.6, 1.15, step=0.01)
        risk_scale = st.slider("風險係數", 0.1, 1.0, 0.3)
        use_mock = st.checkbox("歷史記憶修正", True)
        show_unc = st.toggle("顯示區間", True)

# [MODE 1: 單場預測 (Full Restoration)]
if app_mode == "🎯 單場深度預測":
    st.header("🎯 單場深度預測 (V38 Engine)")
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
    
    t1, t2 = st.tabs(["📋 貼上 JSON", "📂 上傳 JSON"])
    inp = None
    with t1:
        txt = st.text_area("JSON Input", height=100)
        if txt: 
            try: inp = json.loads(txt)
            except: st.error("Format Error")
    with t2:
        f = st.file_uploader("JSON File", type=['json'])
        if f: inp = json.load(f)

    if st.button("🚀 執行分析", type="primary") and inp:
        eng = SniperAnalystLogic(inp, 9, nb_alpha, lam3_in, rho_in, ha_in)
        lh, la, w = eng.calc_lambda()
        M, probs = eng.build_matrix_v38(lh, la, use_biv, use_dc)
        
        bonus = eng.get_market_trend_bonus()
        odds = eng.market["1x2_odds"]
        rid = eng.memory.analyze_scenario(lh, la, odds)
        h_dat = eng.memory.recall_experience(rid)
        penalty = eng.memory.calc_memory_penalty(h_dat["roi"]) if use_mock else 1.0
        
        sens_lv, sens_dr = eng.check_sensitivity(lh, la)
        diff_p = abs(probs["hybrid"]["home"] - probs["market"]["home"])
        conf, reasons = eng.calc_model_confidence(lh, la, diff_p, sens_dr)
        
        hw, dr, aw, sh, sa = eng.run_monte_carlo_vectorized(M)
        
        st.session_state.analysis_results = {
            "eng": eng, "M": M, "lh": lh, "la": la, "w": w,
            "probs": probs, "bonus": bonus, "h_dat": h_dat, "pen": penalty,
            "conf": conf, "reasons": reasons, "sh": sh, "sa": sa
        }

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        eng, M, probs = res["eng"], res["M"], res["probs"]
        
        st.markdown("### 🔍 V38 戰術儀表板")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("主預期", f"{res['lh']:.2f}", delta="加權" if res["w"] else None)
        c2.metric("客預期", f"{res['la']:.2f}")
        c3.metric("混合主勝", f"{probs['hybrid']['home']:.1%}")
        c4.metric("信心", f"{res['conf']:.0%}")
        
        if res["conf"] < 1.0:
            with st.expander("⚠️ 扣分原因"):
                for r in res["reasons"]: st.warning(r)

        t_val, t_ai, t_score, t_sim = st.tabs(["💰 價值投資", "🧠 智能裁決", "🎯 波膽分佈", "🎲 極速模擬"])
        
        candidates = []
        
        # [Tab 1: Value Betting - Detailed Tables]
        with t_val:
            st.subheader("獨贏 (1x2)")
            r_1x2 = []
            for tag, k in [("主勝","home"),("和局","draw"),("客勝","away")]:
                p = probs["hybrid"][k]
                o = eng.market["1x2_odds"][k]
                raw_ev = (p*o - 1)*100 + res["bonus"][k]
                adj_ev = raw_ev * res["conf"] * res["pen"]
                var, sharpe = calc_risk_metrics(p, o)
                kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, p)
                
                ev_str = f"{adj_ev:+.1f}%"
                if show_unc:
                    l, h = eng.simulate_uncertainty(res['lh'], res['la'], adj_ev)
                    ev_str += f" [{l:.1f}, {h:.1f}]"
                
                r_1x2.append({"Pick": tag, "Odds": o, "EV": ev_str, "Kelly": f"{kelly:.1f}%"})
                if adj_ev > 1.0: 
                    candidates.append({"pick": tag, "odds": o, "ev": adj_ev, "kelly": kelly, "type": "1x2"})
            st.dataframe(pd.DataFrame(r_1x2), use_container_width=True)
            
            c_ah, c_ou = st.columns(2)
            with c_ah:
                st.subheader("亞盤 (AH)")
                rows_ah = []
                target = eng.market.get("target_odds", 1.90)
                for hcap in eng.market.get("handicaps", [-0.5, 0.5]):
                    raw = eng.ah_ev(M, hcap, target) + res["bonus"]["home"]
                    adj = raw * res["conf"] * res["pen"]
                    p_approx = (raw/100+1)/target
                    var, _ = calc_risk_metrics(p_approx, target)
                    kel = calc_risk_adj_kelly(adj, var, risk_scale, p_approx)
                    rows_ah.append({"盤口": f"{hcap:+}", "EV": f"{adj:+.1f}%", "Kelly": f"{kel:.1f}%"})
                    if adj > 1.5: candidates.append({"pick":f"AH {hcap:+}", "odds":target, "ev":adj, "kelly":kel, "type":"AH"})
                st.dataframe(pd.DataFrame(rows_ah), use_container_width=True)
                
            with c_ou:
                st.subheader("大小 (OU)")
                rows_ou = []
                idx_sum = np.add.outer(np.arange(eng.max_g), np.arange(eng.max_g))
                for line in eng.market.get("goal_lines", [2.5]):
                    p_over = float(M[idx_sum > line].sum())
                    raw = (p_over*target - 1)*100
                    adj = raw * res["conf"] * res["pen"]
                    var, _ = calc_risk_metrics(p_over, target)
                    kel = calc_risk_adj_kelly(adj, var, risk_scale, p_over)
                    rows_ou.append({"盤口": f"Over {line}", "EV": f"{adj:+.1f}%", "Kelly": f"{kel:.1f}%"})
                    if adj > 1.5: candidates.append({"pick":f"Over {line}", "odds":target, "ev":adj, "kelly":kel, "type":"OU"})
                st.dataframe(pd.DataFrame(rows_ou), use_container_width=True)
                
            st.divider()
            st.markdown("### 🏆 智能投資組合")
            if candidates:
                best = sorted(candidates, key=lambda x: x['ev'], reverse=True)[:3]
                reco = []
                for p in best:
                    amt = unit_stake * (p['kelly']/100)
                    reco.append([f"[{p['type']}] {p['pick']}", p['odds'], f"{p['ev']:+.1f}%", f"{p['kelly']:.1f}%", f"${amt:.1f}"])
                st.dataframe(pd.DataFrame(reco, columns=["選項","賠率","EV","注碼%","金額"]), use_container_width=True)
            else:
                st.info("🚧 風險過高，建議觀望")

        with t_ai:
            st.write("V38 混合權重分析")
            df_c = pd.DataFrame([probs["model"], probs["market"], probs["hybrid"]], index=["Model","Market","Hybrid"])
            st.dataframe(df_c.style.format("{:.1%}"))
            
        with t_score:
            st.write("波膽矩陣")
            st.dataframe(pd.DataFrame(M[:6,:6]).style.format("{:.1%}"))
            
        with t_sim:
            hw = np.sum(res["sh"] > res["sa"]) / 500000
            st.metric("MC 主勝率", f"{hw:.1%}")
            fig, ax = plt.subplots(figsize=(6,2))
            ax.hist(res["sh"], alpha=0.5, label="H"); ax.hist(res["sa"], alpha=0.5, label="A"); ax.legend()
            st.pyplot(fig)
            st.divider()
            st.subheader("稀有事件 (CE-IS)")
            line_chk = 4.5
            ce_res = eng.run_ce_importance_sampling(M, line_chk)
            st.metric(f"大 {line_chk} 機率", f"{ce_res['est']:.2%}")

# [MODE 2: 風險對沖 (Black Text Fix)]
elif app_mode == "🛡️ 風險對沖實驗室":
    st.title("🛡️ 風險對沖")
    if st.session_state.get("analysis_results"):
        res = st.session_state.analysis_results
        sh, sa = res["sh"], res["sa"]
        eng = res["eng"]
        
        if st.button("⚡ 計算組合優化"):
            cands = [
                {"name": "主勝", "odds": eng.market["1x2_odds"]["home"], "cond": (sh > sa)},
                {"name": "和局", "odds": eng.market["1x2_odds"]["draw"], "cond": (sh == sa)},
                {"name": "大2.5", "odds": eng.market.get("target_odds", 1.9), "cond": ((sh+sa) > 2.5)}
            ]
            payoffs = np.zeros((500000, len(cands)))
            for i, c in enumerate(cands): payoffs[:, i] = np.where(c["cond"], c["odds"]-1, -1)
            mu = payoffs.mean(axis=0)
            sigma = np.cov(payoffs, rowvar=False)
            
            def obj(w): return -(np.dot(w, mu) - 2.0 * np.dot(w.T, np.dot(sigma, w)))
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
            opt = minimize(obj, [1/len(cands)]*len(cands), bounds=[(0,1)]*len(cands), constraints=cons)
            
            cols = st.columns(len(cands))
            for i, w in enumerate(opt.x):
                cols[i].metric(cands[i]["name"], f"{w:.1%}", delta=f"EV: {mu[i]*100:.1f}%")
            
            st.markdown("""<div style="background:#f0f2f6; padding:10px; color:#333333; border-radius:5px;">
            <h4 style="margin:0; color:blue;">👨‍🏫 首席分析師評語</h4>
            <p style="color:#333333 !important;">請依照上述比例分配資金以最大化風險回報比 (Sharpe Ratio)。</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("請先執行單場預測")

# [MODE 3: 參數校正 (Multi-File)]
elif app_mode == "🔧 參數校正實驗室":
    st.header("🔧 參數校正 (自動適配)")
    files = st.file_uploader("上傳 CSV/Excel (可多選)", type=['csv','xlsx'], accept_multiple_files=True, key="up_v38_5")
    
    if files:
        dfs = []
        for f in files:
            try:
                if f.name.endswith('.csv'):
                    try: df = pd.read_csv(f, encoding='utf-8')
                    except: f.seek(0); df = pd.read_csv(f, encoding='big5')
                else:
                    import openpyxl; df = pd.read_excel(f)
                
                df = preprocess_uploaded_data(df)
                if not df.empty: dfs.append(df)
            except Exception as e: st.warning(f"{f.name} 失敗: {e}")
            
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            st.write(f"成功合併 {len(dfs)} 個檔案，共 {len(full_df)} 筆數據", full_df.head(3))
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⚡ MLE 擬合"):
                    with st.spinner("計算中..."):
                        r = fit_params_mle(full_df)
                    if r["success"]:
                        st.success(f"建議參數: Lam3={r['lam3']:.3f}, Rho={r['rho']:.3f}, HA={r['home_adv']:.3f}")
                    else: st.error("收斂失敗")
            with c2:
                if st.button("📈 Kalman 追蹤"):
                    h, r = run_kalman_tracking(full_df)
                    st.dataframe(h.tail())

elif app_mode == "📈 聯賽歷史回測":
    st.info("請將 CSV 放入資料夾後使用 Batch Engine")

elif app_mode == "📚 劇本查詢":
    mem = RegimeMemory()
    st.dataframe(pd.DataFrame([{"Name":v["name"], "ROI":v["roi"], "Bets":v["bets"]} for k,v in mem.history_db.items()]))
