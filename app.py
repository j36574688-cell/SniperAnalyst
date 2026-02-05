import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import datetime
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize

# [V40.6] 安全導入 Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# [V38] Numba JIT 加速
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
# 1. 核心數學工具 (Kernel)
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
    G = max_g
    M = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            p = math.exp(biv_poisson_logpmf_fast(i, j, lh, la, 0.0))
            M[i, j] = p
    return M / M.sum()

# =========================
# 2. 全景記憶與實戰系統
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

class PaperTradingSystem:
    def __init__(self, file_path="my_bets.csv"):
        self.file_path = file_path
        
    def load_bets(self):
        if os.path.exists(self.file_path):
            try:
                return pd.read_csv(self.file_path)
            except:
                return pd.DataFrame(columns=["Date", "Selection", "Odds", "Stake", "Result", "PnL"])
        return pd.DataFrame(columns=["Date", "Selection", "Odds", "Stake", "Result", "PnL"])
        
    def add_bet(self, selection, odds, stake):
        df = self.load_bets()
        new_row = pd.DataFrame([{
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Selection": selection,
            "Odds": odds,
            "Stake": stake,
            "Result": "Pending",
            "PnL": 0.0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.file_path, index=False)
        return True
    
    def save_bets(self, df):
        # 自動重新計算 PnL
        for idx, row in df.iterrows():
            res = row['Result']
            stake = float(row['Stake'])
            odds = float(row['Odds'])
            if res == "Win": df.at[idx, 'PnL'] = stake * (odds - 1)
            elif res == "Lose": df.at[idx, 'PnL'] = -stake
            elif res == "Void": df.at[idx, 'PnL'] = 0.0
            else: df.at[idx, 'PnL'] = 0.0
        df.to_csv(self.file_path, index=False)
    
    def get_stats(self):
        df = self.load_bets()
        if df.empty: return 0, 0, 0
        total_bets = len(df)
        total_stake = df["Stake"].sum()
        total_pnl = df["PnL"].sum()
        return total_bets, total_stake, total_pnl

# =========================
# 3. 分析引擎邏輯
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
        
        lh = (lh_att * la_def / 1.35) * self.home_adv * crush_factor
        la = (la_att * lh_def / 1.35)
        
        if self.h["context_modifiers"].get("missing_key_defender"): lh *= 0.9 
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
        bonus = {"home":0.0, "draw":0.0, "away":0.0}
        op, cu = self.market.get("opening_odds"), self.market.get("1x2_odds")
        if not op or not cu: return bonus
        for k in bonus:
            drop = max(0.0, (op[k] - cu[k]) / op[k])
            bonus[k] = min(3.0, drop * 30.0)
        return bonus

    def ah_ev(self, M, hcap, odds):
        q = int(round(hcap * 4))
        if q % 2 != 0: return 0.5 * self.ah_ev(M, (q+1)/4.0, odds) + 0.5 * self.ah_ev(M, (q-1)/4.0, odds)
        idx_diff = np.subtract.outer(np.arange(self.max_g), np.arange(self.max_g)) 
        payoff = np.select([idx_diff + hcap > 0.001, np.abs(idx_diff + hcap) <= 0.001], [odds-1, 0], default=-1)
        return np.sum(M * payoff) * 100

    def check_sensitivity(self, lh, la):
        M_stress = get_matrix_cached(lh, la + 0.3, self.max_g, self.nb_alpha)
        p_orig = float(np.sum(np.tril(get_matrix_cached(lh, la, self.max_g, self.nb_alpha), -1)))
        p_new = float(np.sum(np.tril(M_stress, -1)))
        drop = (p_orig - p_new) / p_orig if p_orig > 0 else 0
        return ("High" if drop > 0.15 else "Medium"), drop

    def calc_model_confidence(self, lh, la, diff, sens):
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
        i_idx, j_idx = np.indices((G,G))
        mu_h = np.sum(M * i_idx)
        mu_a = np.sum(M * j_idx)
        v_h, v_a = mu_h * 1.5, mu_a * 1.5
        rng = np.random.default_rng()
        sh = rng.poisson(v_h, n_sims)
        sa = rng.poisson(v_a, n_sims)
        log_w = (sh*(np.log(mu_h)-np.log(v_h)) - (mu_h-v_h)) + \
                (sa*(np.log(mu_a)-np.log(v_a)) - (mu_a-v_a))
        w = np.exp(log_w)
        est = np.sum(w * ((sh+sa)>line)) / n_sims
        return {"est": float(est)}

# =========================
# 4. 資料處理工具 (V40.3 強力讀取版)
# =========================
def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {
        'hometeam': 'home', 'home': 'home', 'ht': 'home', 'team1': 'home',
        'awayteam': 'away', 'away': 'away', 'at': 'away', 'team2': 'away',
        'fthg': 'home_goals', 'hg': 'home_goals', 'homegoals': 'home_goals', 'score1': 'home_goals',
        'ftag': 'away_goals', 'ag': 'away_goals', 'awaygoals': 'away_goals', 'score2': 'away_goals',
        'div': 'div', 'date': 'date'
    }
    new_cols = {}
    for col in df.columns:
        c_lower = col.lower().replace(" ", "").replace("_", "")
        if c_lower in col_map: new_cols[col] = col_map[c_lower]
    df = df.rename(columns=new_cols)
    required = ['home', 'away', 'home_goals', 'away_goals']
    if any(c not in df.columns for c in required): return pd.DataFrame()
    if 'lh_pred' not in df.columns or 'la_pred' not in df.columns:
        avg_h = df['home_goals'].mean()
        avg_a = df['away_goals'].mean()
        df['lh_pred'] = avg_h; df['la_pred'] = avg_a
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

# [V39/40 視覺化工具]
def plot_score_heatmap(M):
    if not HAS_PLOTLY: return None
    limit = 6
    labels = [str(i) for i in range(limit)]
    fig = px.imshow(M[:limit, :limit], 
                    labels=dict(x="客隊進球", y="主隊進球", color="機率"),
                    x=labels, y=labels, text_auto='.1%')
    fig.update_layout(title="波膽機率熱力圖", width=500, height=400)
    return fig

def plot_sensitivity_surface(lh_base, la_base, lam3, rho, max_g):
    if not HAS_PLOTLY: return None
    x = np.linspace(0.8, 1.2, 10)
    y = np.linspace(0.8, 1.2, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(10):
        for j in range(10):
            l1, l2 = lh_base * X[i,j], la_base * Y[i,j]
            p = 0
            for h in range(max_g):
                for a in range(h):
                    p += math.exp(biv_poisson_logpmf_fast(h, a, max(0.01, l1-lam3), max(0.01, l2-lam3), lam3))
            Z[i,j] = p
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title="主勝機率敏感度", scene=dict(xaxis_title="主隊係數", yaxis_title="客隊係數", zaxis_title="主勝率"))
    return fig

def plot_radar_chart(lh, la):
    if not HAS_PLOTLY: return None
    def normalize(val): return min(100, max(20, val * 40))
    categories = ['進攻能力', '防守壓迫', '近期狀態', '主客優勢', '運氣指數']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[normalize(lh), normalize(1/la), 75, 80, 50], theta=categories, fill='toself', name='主隊'))
    fig.add_trace(go.Scatterpolar(r=[normalize(la), normalize(1/lh), 65, 40, 50], theta=categories, fill='toself', name='客隊'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="球隊戰力雷達")
    return fig

def plot_calendar_heatmap(df_bets):
    if not HAS_PLOTLY or df_bets.empty: return None
    if "Date" not in df_bets.columns or "PnL" not in df_bets.columns: return None
    df_bets['DateObj'] = pd.to_datetime(df_bets['Date']).dt.date
    daily = df_bets.groupby('DateObj')['PnL'].sum().reset_index()
    fig = px.density_heatmap(daily, x="DateObj", y="PnL", title="獲利日曆熱力圖", nbinsx=20)
    return fig

# =========================
# 5. UI (V40.6 Grand Fix)
# =========================
st.set_page_config(page_title="Sniper V40.6", page_icon="🧿", layout="wide")
st.markdown("<style>.metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; } .stProgress > div > div > div > div { background-color: #4CAF50; }</style>", unsafe_allow_html=True)

# 初始化
ptrader = PaperTradingSystem()
if "cart" not in st.session_state: st.session_state.cart = []

with st.sidebar:
    st.title("🧿 Sniper V40.6")
    st.caption("Grand Fix Edition")
    if HAS_NUMBA: st.success("⚡ Numba 加速：已啟動")
    else: st.warning("⚠️ Numba 加速：未啟動")
    
    # 戰情室
    n_bets, t_stake, t_pnl = ptrader.get_stats()
    st.markdown("### 🏎️ 戰情室")
    col_w1, col_w2 = st.columns(2)
    col_w1.metric("模擬本金", "$10,000")
    col_w2.metric("累積損益", f"${t_pnl:.1f}", delta=f"{t_pnl/100:.1f}%")
    st.metric("今日注單 / 總額", f"{len(st.session_state.cart)} / {n_bets}", f"${t_stake:.0f}")
    
    st.divider()
    app_mode = st.radio("功能模式：", ["🎯 單場深度預測", "🛡️ 風險對沖實驗室", "🔧 參數校正實驗室", "📈 實戰績效回顧", "📚 劇本查詢"])
    st.divider()
    
    # 購物車
    with st.expander(f"🛒 待確認注單 ({len(st.session_state.cart)})", expanded=False):
        if st.session_state.cart:
            for i, bet in enumerate(st.session_state.cart):
                st.write(f"{i+1}. {bet['sel']} @ {bet['odds']} (${bet['stake']})")
            if st.button("✅ 一鍵下注"):
                for bet in st.session_state.cart: ptrader.add_bet(bet['sel'], bet['odds'], bet['stake'])
                st.session_state.cart = []
                st.success("下注成功！")
                st.rerun()
            if st.button("🗑️ 清空"):
                st.session_state.cart = []
                st.rerun()
        else: st.info("暫無注單")

    with st.expander("🛠️ 進階參數", expanded=False):
        unit_stake = st.number_input("單注本金", 10, 10000, 100)
        nb_alpha = st.slider("Alpha", 0.05, 0.25, 0.12)
        use_biv = st.toggle("雙變量", True)
        use_dc = st.toggle("Dixon-Coles", True)
        lam3_in = st.number_input("Lambda 3", 0.0, 0.5, 0.15)
        rho_in = st.number_input("Rho", -0.3, 0.3, -0.13)
        ha_in = st.number_input("Home Adv", 0.8, 1.6, 1.15)
        risk_scale = st.slider("Kelly 係數", 0.1, 1.0, 0.3)
        show_unc = st.toggle("顯示區間", True)

if app_mode == "🎯 單場深度預測":
    st.header("🎯 單場深度預測 (V40)")
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
    
    t1, t2 = st.tabs(["📋 貼上 JSON", "📂 上傳 JSON"])
    inp = None
    with t1:
        txt = st.text_area("JSON Input", height=100)
        if txt: 
            try: inp = json.loads(txt)
            except: st.error("Error")
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
        sens_lv, sens_dr = eng.check_sensitivity(lh, la)
        diff_p = abs(probs["hybrid"]["home"] - probs["market"]["home"])
        conf, reasons = eng.calc_model_confidence(lh, la, diff_p, sens_dr)
        hw, dr, aw, sh, sa = eng.run_monte_carlo_vectorized(M)
        
        st.session_state.analysis_results = {
            "eng": eng, "M": M, "lh": lh, "la": la, "w": w,
            "probs": probs, "bonus": bonus, "h_dat": h_dat, "pen": 1.0,
            "conf": conf, "reasons": reasons, "sh": sh, "sa": sa
        }

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        eng, M, probs = res["eng"], res["M"], res["probs"]
        
        st.markdown("### 🔍 戰術儀表板")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("主隊預期", f"{res['lh']:.2f}")
        c2.metric("客隊預期", f"{res['la']:.2f}")
        c3.metric("模型主勝", f"{probs['hybrid']['home']:.1%}")
        c4.metric("信心", f"{res['conf']:.0%}")

        t_val, t_ai, t_vis, t_sim, t_sand = st.tabs(["💰 價值投資", "🧠 智能裁決", "🌈 視覺洞察", "🎲 極速模擬", "🔮 終極沙盤推演"])
        
        candidates = []
        with t_val:
            st.subheader("獨贏 (1x2)")
            r_1x2 = []
            for tag, k in [("主勝","home"),("和局","draw"),("客勝","away")]:
                p = probs["hybrid"][k]
                o = eng.market["1x2_odds"][k]
                raw_ev = (p*o - 1)*100 + res["bonus"][k]
                adj_ev = raw_ev * res["conf"]
                var, sharpe = calc_risk_metrics(p, o)
                kelly = calc_risk_adj_kelly(adj_ev, var, risk_scale, p)
                amt = unit_stake * (kelly/100.0)
                r_1x2.append({"選項": tag, "賠率": o, "機率": f"{p:.1%}", "期望值": f"{adj_ev:.1f}%", "凱利建議": f"{kelly:.1f}%", "建議金額": f"${amt:.0f}"})
                if adj_ev > 0.2: 
                    candidates.append({"pick": tag, "odds": o, "ev": adj_ev, "kelly": kelly, "type": "1x2", "prob": p, "sharpe": sharpe})
            st.dataframe(pd.DataFrame(r_1x2), use_container_width=True)
            
            # [V40.6] 亞盤優化 - 顯示「誰讓分」
            c_ah, c_ou = st.columns(2)
            with c_ah:
                st.subheader("亞盤 (AH)")
                rows_ah = []
                target = eng.market.get("target_odds", 1.90)
                for hcap in eng.market.get("handicaps", [-0.5, 0.5]):
                    raw = eng.ah_ev(M, hcap, target) + res["bonus"]["home"]
                    adj = raw * res["conf"]
                    p_approx = (raw/100+1)/target
                    var, sharpe = calc_risk_metrics(p_approx, target)
                    kel = calc_risk_adj_kelly(adj, var, risk_scale, p_approx)
                    amt = unit_stake * (kel/100.0)
                    
                    # 判斷讓分方
                    if hcap < 0: tag_str = f"主讓 {hcap}"
                    elif hcap > 0: tag_str = f"主受 +{hcap}"
                    else: tag_str = "平手盤"
                    
                    rows_ah.append({"盤口": tag_str, "機率": f"{p_approx:.1%}", "期望值": f"{adj:.1f}%", "凱利": f"{kel:.1f}%", "金額": f"${amt:.0f}"})
                    if adj > 0.5: candidates.append({"pick": tag_str, "odds": target, "ev": adj, "kelly": kel, "type": "AH", "prob": p_approx, "sharpe": sharpe})
                st.dataframe(pd.DataFrame(rows_ah), use_container_width=True)
            
            with c_ou:
                st.subheader("大小 (OU)")
                rows_ou = []
                idx_sum = np.add.outer(np.arange(eng.max_g), np.arange(eng.max_g))
                for line in eng.market.get("goal_lines", [2.5]):
                    p_over = float(M[idx_sum > line].sum())
                    raw = (p_over*target - 1)*100
                    adj = raw * res["conf"]
                    var, sharpe = calc_risk_metrics(p_over, target)
                    kel = calc_risk_adj_kelly(adj, var, risk_scale, p_over)
                    amt = unit_stake * (kel/100.0)
                    rows_ou.append({"盤口": f"大 {line}", "機率": f"{p_over:.1%}", "期望值": f"{adj:.1f}%", "凱利": f"{kel:.1f}%", "金額": f"${amt:.0f}"})
                    if adj > 0.5: candidates.append({"pick":f"大 {line}", "odds":target, "ev":adj, "kelly":kel, "type":"OU", "prob": p_over, "sharpe": sharpe})
                st.dataframe(pd.DataFrame(rows_ou), use_container_width=True)
                
            st.divider()
            st.markdown("### 🏆 智能投資組合")
            if candidates:
                best = sorted(candidates, key=lambda x: x['ev'], reverse=True)
                reco = []
                for p in best:
                    amt = unit_stake * (p['kelly']/100)
                    reco.append({"選項": f"[{p['type']}] {p['pick']}", "賠率": p['odds'], "期望值": f"{p['ev']:+.1f}%", "凱利%": f"{p['kelly']:.1f}%", "建議$": f"${amt:.1f}"})
                st.dataframe(pd.DataFrame(reco), use_container_width=True)
                
                c_cart1, c_cart2 = st.columns([3, 1])
                bet_pick = c_cart1.selectbox("加入購物車", [f"[{p['type']}] {p['pick']}" for p in best])
                if c_cart2.button("➕"):
                    sel = next(p for p in best if f"[{p['type']}] {p['pick']}" == bet_pick)
                    amt = unit_stake * (sel['kelly']/100)
                    st.session_state.cart.append({"sel": bet_pick, "odds": sel['odds'], "stake": amt})
                    st.success("已加入")
                    st.rerun()
            else: st.info("無推薦注單")

        with t_ai:
            st.write("權重分析")
            st.dataframe(pd.DataFrame([probs["model"], probs["market"], probs["hybrid"]], index=["純模型","市場隱含","混合權重"]))

        with t_vis:
            st.subheader("🌈 視覺洞察")
            if HAS_PLOTLY:
                st.plotly_chart(plot_radar_chart(res['lh'], res['la']), use_container_width=True)
                st.divider()
                c_v1, c_v2 = st.columns(2)
                with c_v1: st.plotly_chart(plot_score_heatmap(M), use_container_width=True)
                with c_v2: st.plotly_chart(px.histogram(x=res["sh"], nbins=10, title="主隊進球"), use_container_width=True)
                st.plotly_chart(plot_sensitivity_surface(res['lh'], res['la'], lam3_in, rho_in, 9), use_container_width=True)
            else: st.warning("請安裝 Plotly")

        with t_sim:
            hw = np.sum(res["sh"] > res["sa"]) / 500000
            st.metric("MC 主勝", f"{hw:.1%}")
            ce_res = eng.run_ce_importance_sampling(M, 4.5)
            st.metric("大 4.5 機率", f"{ce_res['est']:.2%}")

        with t_sand:
            st.subheader("🔮 全域沙盤推演")
            st.info("調整參數，即時預覽變化。")
            sc1, sc2, sc3 = st.columns(3)
            mod_ah = sc1.slider("主隊進攻", 0.5, 1.5, 1.0, 0.05)
            mod_da = sc1.slider("客隊防守", 0.5, 1.5, 1.0, 0.05)
            mod_aa = sc2.slider("客隊進攻", 0.5, 1.5, 1.0, 0.05)
            mod_dh = sc2.slider("主隊防守", 0.5, 1.5, 1.0, 0.05)
            luck = sc3.slider("運氣偏差", 0.8, 1.2, 1.0, 0.05)
            red = sc3.checkbox("主隊紅牌")
            
            lh_n = res['lh'] * mod_ah * mod_da * luck
            la_n = res['la'] * mod_aa * mod_dh * luck
            if red: lh_n *= 0.4; la_n *= 1.3
            
            st.write(f"調整後: 主 {lh_n:.2f} | 客 {la_n:.2f}")
            M_n, _ = eng.build_matrix_v38(lh_n, la_n, use_biv, use_dc)
            ph_n = float(np.sum(np.tril(M_n,-1)))
            
            c_r1, c_r2 = st.columns(2)
            c_r1.metric("新主勝率", f"{ph_n:.1%}")
            o_h = eng.market["1x2_odds"]["home"]
            nev = (ph_n*o_h-1)*100
            c_r2.metric("新 EV", f"{nev:.1f}%", delta_color="normal" if nev>0 else "inverse")

elif app_mode == "🛡️ 風險對沖實驗室":
    st.title("🛡️ 風險對沖實驗室")
    # [V40.6] 恢復完整功能與中文化
    tab_arb, tab_lay, tab_port = st.tabs(["⚡ 1x2 套利", "📉 交易所對沖", "📊 智能組合優化"])
    
    with tab_arb:
        c1, c2, c3 = st.columns(3)
        o1 = c1.number_input("主勝賠率", 2.0); o2 = c2.number_input("和局賠率", 3.0); o3 = c3.number_input("客勝賠率", 4.0)
        inv = 1/o1+1/o2+1/o3
        if inv<1: st.success(f"發現套利機會! ROI: {1/inv-1:.1%}")
        else: st.info(f"無套利空間 (Book: {inv:.2%})")

    with tab_lay:
        c1, c2 = st.columns(2)
        b_o = c1.number_input("Back 賠率", 1.01, 10.0, 2.5)
        stake = c1.number_input("Back 本金", 10, 1000, 100)
        l_o = c2.number_input("Lay 賠率", 1.01, 10.0, 2.6)
        comm = c2.number_input("佣金 %", 0.0, 5.0, 2.0)/100
        if l_o>1:
            lay_s = (stake*b_o)/(l_o-comm)
            st.metric("建議 Lay 金額", f"${lay_s:.2f}")

    with tab_port:
        if st.session_state.get("analysis_results"):
            res = st.session_state.analysis_results
            sh, sa = res["sh"], res["sa"]
            eng = res["eng"]
            if st.button("⚡ 計算最佳配置"):
                cands = [{"name":"主勝","odds":eng.market["1x2_odds"]["home"],"cond":sh>sa}, {"name":"和局","odds":eng.market["1x2_odds"]["draw"],"cond":sh==sa}, {"name":"大2.5","odds":1.9,"cond":(sh+sa)>2.5}]
                pay = np.zeros((500000,3))
                for i,c in enumerate(cands): pay[:,i] = np.where(c["cond"], c["odds"]-1, -1)
                mu, sigma = pay.mean(axis=0), np.cov(pay, rowvar=False)
                cons = ({'type':'eq','fun':lambda w: sum(w)-1})
                opt = minimize(lambda w: -(np.dot(w,mu)-np.dot(w.T,np.dot(sigma,w))), [0.33]*3, bounds=[(0,1)]*3, constraints=cons)
                for i,w in enumerate(opt.x): st.metric(cands[i]["name"], f"{w:.1%}")
                
                ret = np.dot(opt.x, mu)*100
                st.markdown(f"""<div style='background:#f0f2f6;padding:10px;color:black'>
                <b>首席分析師:</b> 預期回報 {ret:.2f}%。建議 {"分散配置以降低波動" if max(opt.x)<0.7 else "集中單打高價值選項"}。</div>""", unsafe_allow_html=True)
        else: st.warning("請先執行單場預測")

elif app_mode == "🔧 參數校正實驗室":
    st.header("🔧 參數校正")
    files = st.file_uploader("上傳 CSV/Excel", type=['csv','xlsx'], accept_multiple_files=True)
    if files:
        dfs = [preprocess_uploaded_data(pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)) for f in files]
        full = pd.concat([d for d in dfs if not d.empty])
        if st.button("⚡ MLE"):
            r = fit_params_mle(full)
            if r["success"]: st.success(f"Lam3={r['lam3']:.2f}, Rho={r['rho']:.2f}, HA={r['home_adv']:.2f}")

# [MODE 4: 實戰績效回顧 (Fixed Crash)]
elif app_mode == "📈 實戰績效回顧":
    st.title("📈 實戰績效回顧")
    df = ptrader.load_bets()
    
    if not df.empty:
        st.markdown("### 📝 注單管理 (直接點擊表格修改)")
        # [V40.6 Fix] Corrected SelectboxColumn
        edited_df = st.data_editor(
            df,
            column_config={
                "Result": st.column_config.SelectboxColumn(
                    "比賽結果",
                    width="medium",
                    options=["Pending", "Win", "Lose", "Void"],
                    required=True,
                ),
                "PnL": st.column_config.NumberColumn(
                    "損益 (PnL)",
                    format="$%.1f",
                    disabled=True 
                )
            },
            num_rows="dynamic",
            use_container_width=True
        )
        
        if st.button("💾 保存變更 & 結算損益"):
            ptrader.save_bets(edited_df)
            st.success("已更新損益狀態！")
            st.rerun()
            
        st.divider()
        if HAS_PLOTLY and "PnL" in df.columns:
            st.subheader("💰 資金成長曲線")
            df["CumPnL"] = df["PnL"].cumsum()
            st.plotly_chart(px.line(df, x="Date", y="CumPnL", markers=True), use_container_width=True)
            st.subheader("📅 獲利日曆")
            st.plotly_chart(plot_calendar_heatmap(df), use_container_width=True)
    else:
        st.info("尚無模擬注單。請在「單場深度預測」中加入注單。")

elif app_mode == "📚 劇本查詢":
    st.dataframe(pd.DataFrame([{"Name":v["name"],"ROI":v["roi"]} for k,v in RegimeMemory().history_db.items()]))
