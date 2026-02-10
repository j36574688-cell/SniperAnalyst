import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import datetime
import sqlite3
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize

# [V41.1] 安全導入 Plotly
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
# 2. [V42.0] 用戶系統與資料庫核心
# =========================
class AuthManager:
    def __init__(self, db_path="sniper_v42.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """初始化資料庫與資料表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # 用戶表
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY, 
                        password_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_admin INTEGER DEFAULT 0)''')  # Added is_admin field
        # 注單表 (新增 user_id)
        c.execute('''CREATE TABLE IF NOT EXISTS bets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        date TEXT,
                        selection TEXT,
                        odds REAL,
                        stake REAL,
                        result TEXT,
                        pnl REAL,
                        FOREIGN KEY(user_id) REFERENCES users(username))''')
        
        # Check if admin user exists, if not create one
        # Default admin: admin / admin123 (You should change this immediately)
        c.execute("SELECT * FROM users WHERE username='admin'")
        if not c.fetchone():
             admin_pass = self.hash_password("admin123")
             c.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)", ("admin", admin_pass, 1))
             conn.commit()

        conn.commit()
        conn.close()

    def hash_password(self, password):
        # 使用 SHA256 + Salt (簡單有效的加密)
        salt = "SniperTarget" 
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def register(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            hashed = self.hash_password(password)
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed))
            conn.commit()
            return True, "註冊成功！請登入。"
        except sqlite3.IntegrityError:
            return False, "用戶名已存在。"
        finally:
            conn.close()

    def login(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        hashed = self.hash_password(password)
        c.execute("SELECT * FROM users WHERE username=? AND password_hash=?", (username, hashed))
        user = c.fetchone()
        conn.close()
        # user structure: (username, password_hash, created_at, is_admin)
        if user:
             return True, user[3] # Return success and is_admin status
        return False, 0

    # New Admin Function: Get all users
    def get_all_users(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT username, created_at, is_admin FROM users", conn)
        conn.close()
        return df
    
    # New Admin Function: Delete user
    def delete_user(self, username):
        if username == 'admin': return False # Prevent deleting super admin
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username=?", (username,))
        c.execute("DELETE FROM bets WHERE user_id=?", (username,)) # Cascade delete bets
        conn.commit()
        conn.close()
        return True

class PaperTradingSystemSQL:
    def __init__(self, user_id, db_path="sniper_v42.db"):
        self.db_path = db_path
        self.user_id = user_id # 鎖定當前用戶

    def load_bets(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM bets WHERE user_id = ?", conn, params=(self.user_id,))
        conn.close()
        return df

    def add_bet(self, selection, odds, stake):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        c.execute("INSERT INTO bets (user_id, date, selection, odds, stake, result, pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (self.user_id, date_str, selection, odds, stake, "Pending", 0.0))
        conn.commit()
        conn.close()

    def save_bets(self, df):
        # This function updates bets based on the edited DataFrame
        # It's crucial to match by ID to ensure correct updates
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for index, row in df.iterrows():
            bet_id = row['id']
            result = row['Result']
            stake = float(row['Stake'])
            odds = float(row['Odds'])
            
            # Recalculate PnL based on result
            pnl = 0.0
            if result == "Win":
                pnl = stake * (odds - 1)
            elif result == "Lose":
                pnl = -stake
            elif result == "Void":
                pnl = 0.0
            
            # Update the record in database
            c.execute("UPDATE bets SET result=?, pnl=? WHERE id=? AND user_id=?", 
                      (result, pnl, bet_id, self.user_id))
            
        conn.commit()
        conn.close()

    def get_stats(self):
        df = self.load_bets()
        if df.empty: return 0, 0, 0
        return len(df), df["Stake"].sum(), df["PnL"].sum()

# =========================
# 3. 分析引擎邏輯 (Kernel)
# =========================
class RegimeMemory:
    def __init__(self):
        self.history_db = {
            "Bore_Draw_Stalemate": { "name": "🛡️ 雙重鐵桶", "roi": 0.219, "bets": 2150 }, 
            "Relegation_Dog": { "name": "🐕 保級受讓", "roi": 0.083, "bets": 1840 },
            "Fallen_Giant": { "name": "📉 豪門崩盤", "roi": -0.008, "bets": 920 },
            "Fortress_Home": { "name": "🏰 魔鬼主場", "roi": -0.008, "bets": 3100 },
            "Title_MustWin_Home": { "name": "🏆 爭冠必勝盤", "roi": -0.063, "bets": 2450 },
            "MarketHype_Fav": { "name": "🔥 大熱倒灶", "roi": -0.080, "bets": 1560 },
            "MidTable_Standard": { "name": "😐 中游例行", "roi": 0.000, "bets": 5000 }
        }
    def analyze_scenario(self, lh, la, odds) -> str:
        h = odds.get("home", 2.0)
        if h < 1.30: return "MarketHype_Fav"
        if (lh+la) < 2.2: return "Bore_Draw_Stalemate"
        return "MidTable_Standard"
    def recall_experience(self, rid: str) -> Dict:
        return self.history_db.get(rid, {"name": "未知", "roi": 0.0, "bets": 0})

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
        imp = {"home": 1/self.market["1x2_odds"]["home"], "draw": 1/self.market["1x2_odds"]["draw"], "away": 1/self.market["1x2_odds"]["away"]}
        total_imp = sum(imp.values())
        imp = {k: v/total_imp for k, v in imp.items()}
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

    def ah_ev(self, M, hcap, odds):
        q = int(round(hcap * 4))
        if q % 2 != 0: return 0.5 * self.ah_ev(M, (q+1)/4.0, odds) + 0.5 * self.ah_ev(M, (q-1)/4.0, odds)
        idx_diff = np.subtract.outer(np.arange(self.max_g), np.arange(self.max_g)) 
        r_matrix = idx_diff + hcap
        payoff = np.select([r_matrix > 0.001, np.abs(r_matrix) <= 0.001, r_matrix < -0.001], [odds - 1, 0, -1], default=-1)
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

def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {'hometeam': 'home', 'home': 'home', 'awayteam': 'away', 'away': 'away', 'fthg': 'home_goals', 'ftag': 'away_goals'}
    new_cols = {}
    for col in df.columns:
        c_lower = col.lower().replace(" ", "").replace("_", "")
        if c_lower in col_map: new_cols[col] = col_map[c_lower]
    df = df.rename(columns=new_cols)
    required = ['home', 'away', 'home_goals', 'away_goals']
    if any(c not in df.columns for c in required): return pd.DataFrame()
    if 'lh_pred' not in df.columns:
        avg_h = df['home_goals'].mean()
        avg_a = df['away_goals'].mean()
        df['lh_pred'] = avg_h
        df['la_pred'] = avg_a
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

# [V39/40 視覺化]
def plot_score_heatmap(M):
    if not HAS_PLOTLY: return None
    limit = 6
    labels = [str(i) for i in range(limit)]
    fig = px.imshow(M[:limit, :limit], labels=dict(x="客隊", y="主隊", color="機率"), x=labels, y=labels, text_auto='.1%')
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
    fig.update_layout(title="主勝敏感度", scene=dict(xaxis_title="主隊", yaxis_title="客隊", zaxis_title="勝率"))
    return fig

def plot_radar_chart(lh, la):
    if not HAS_PLOTLY: return None
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[min(100, lh*40), min(100, 1/la*40), 75, 80], theta=['進攻', '防守', '近況', '主客'], fill='toself', name='主隊'))
    fig.add_trace(go.Scatterpolar(r=[min(100, la*40), min(100, 1/lh*40), 65, 40], theta=['進攻', '防守', '近況', '主客'], fill='toself', name='客隊'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="戰力雷達")
    return fig

def plot_calendar_heatmap(df_bets):
    if not HAS_PLOTLY or df_bets.empty: return None
    if "date" not in df_bets.columns or "pnl" not in df_bets.columns: return None
    df_bets['DateObj'] = pd.to_datetime(df_bets['date']).dt.date
    daily = df_bets.groupby('DateObj')['pnl'].sum().reset_index()
    fig = px.density_heatmap(daily, x="DateObj", y="pnl", title="獲利日曆", nbinsx=20)
    return fig

# =========================
# 6. 主程式架構 (Platform)
# =========================
st.set_page_config(page_title="Sniper V42.0", page_icon="🧿", layout="wide")
st.markdown("<style>.metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; } .stProgress > div > div > div > div { background-color: #4CAF50; }</style>", unsafe_allow_html=True)

# 初始化
auth = AuthManager()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.is_admin = 0

# --- 登入/註冊頁面 (Gatekeeper) ---
if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("🔐 Sniper V42.0 戰情室")
        tab_login, tab_reg = st.tabs(["登入", "註冊新帳號"])
        
        with tab_login:
            u = st.text_input("帳號", key="l_u")
            p = st.text_input("密碼", type="password", key="l_p")
            if st.button("登入"):
                success, is_admin = auth.login(u, p)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.is_admin = is_admin
                    st.rerun()
                else:
                    st.error("帳號或密碼錯誤")
        
        with tab_reg:
            nu = st.text_input("設定帳號", key="r_u")
            np1 = st.text_input("設定密碼", type="password", key="r_p1")
            np2 = st.text_input("確認密碼", type="password", key="r_p2")
            if st.button("註冊"):
                if np1 != np2:
                    st.error("兩次密碼不符")
                elif len(nu) < 3:
                    st.error("帳號太短")
                else:
                    success, msg = auth.register(nu, np1)
                    if success: st.success(msg)
                    else: st.error(msg)

# --- 主程式 (Logged In) ---
else:
    # 初始化該用戶的交易系統 (SQL)
    ptrader = PaperTradingSystemSQL(st.session_state.username)
    if "cart" not in st.session_state: st.session_state.cart = []

    with st.sidebar:
        st.title(f"👮‍♂️ {st.session_state.username}")
        if st.session_state.is_admin:
            st.success("👑 管理員權限已啟用")
        
        if st.button("🚪 登出"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.is_admin = 0
            st.rerun()
        
        st.divider()
        n_bets, t_stake, t_pnl = ptrader.get_stats()
        st.markdown("### 🏎️ 戰情室")
        c1, c2 = st.columns(2)
        c1.metric("本金", "$10,000")
        c2.metric("損益", f"${t_pnl:.1f}", delta=f"{t_pnl/100:.1f}%")
        st.metric("今日", f"{len(st.session_state.cart)} / {n_bets}", f"${t_stake:.0f}")
        
        st.divider()
        
        # 動態調整選單：管理員多一個「用戶管理」
        menu_options = ["🎯 單場深度預測", "🛡️ 風險對沖實驗室", "🔧 參數校正實驗室", "📈 實戰績效回顧", "📚 劇本查詢"]
        if st.session_state.is_admin:
            menu_options.append("👑 用戶管理 (Admin)")
            
        app_mode = st.radio("模式", menu_options)
        
        with st.expander(f"🛒 購物車 ({len(st.session_state.cart)})"):
            if st.session_state.cart:
                for i, b in enumerate(st.session_state.cart): st.write(f"{i+1}. {b['sel']} @ {b['odds']} (${b['stake']:.0f})")
                if st.button("✅ 下注"):
                    for b in st.session_state.cart: ptrader.add_bet(b['sel'], b['odds'], b['stake'])
                    st.session_state.cart = []
                    st.success("OK")
                    st.rerun()
                if st.button("🗑️ 清空"):
                    st.session_state.cart = []
                    st.rerun()
            else: st.info("空")

        with st.expander("🛠️ 參數"):
            unit_stake = st.number_input("單注", 10, 10000, 100)
            nb_alpha = st.slider("Alpha", 0.05, 0.25, 0.12)
            use_biv = st.toggle("雙變量", True)
            use_dc = st.toggle("DC修正", True)
            lam3_in = st.number_input("Lam3", 0.0, 0.5, 0.15)
            rho_in = st.number_input("Rho", -0.3, 0.3, -0.13)
            ha_in = st.number_input("主場優勢", 0.8, 1.6, 1.15)
            risk_scale = st.slider("Kelly", 0.1, 1.0, 0.3)

    if app_mode == "🎯 單場深度預測":
        st.header("🎯 單場深度預測 (V42 SQL)")
        t1, t2 = st.tabs(["JSON 文字", "JSON 檔案"])
        inp = None
        with t1:
            txt = st.text_area("Input JSON", height=100)
            if txt: 
                try: inp = json.loads(txt)
                except: st.error("Error")
        with t2:
            f = st.file_uploader("Upload JSON", type=['json'])
            if f: inp = json.load(f)

        if st.button("🚀 分析") and inp:
            eng = SniperAnalystLogic(inp, 9, nb_alpha, lam3_in, rho_in, ha_in)
            lh, la, w = eng.calc_lambda()
            M, probs = eng.build_matrix_v38(lh, la, use_biv, use_dc)
            conf, reasons = eng.calc_model_confidence(lh, la, 0.1, 0.0)
            hw, dr, aw, sh, sa = eng.run_monte_carlo_vectorized(M)
            st.session_state.res = {"eng": eng, "M": M, "lh": lh, "la": la, "probs": probs, "conf": conf, "sh": sh, "sa": sa}

        if "res" in st.session_state and st.session_state.res:
            res = st.session_state.res
            eng, M = res["eng"], res["M"]
            st.markdown("### 📊 儀表板")
            c1, c2, c3 = st.columns(3)
            c1.metric("主預期", f"{res['lh']:.2f}")
            c2.metric("客預期", f"{res['la']:.2f}")
            c3.metric("信心", f"{res['conf']:.0%}")
            
            t_v, t_a, t_vis, t_sim, t_sand = st.tabs(["💰 價值", "🧠 智能", "🌈 視覺", "🎲 模擬", "🔮 沙盤"])
            
            best_bets = []
            with t_v:
                # 1x2 Table
                r1x2 = []
                for t, k in [("主勝","home"),("和","draw"),("客勝","away")]:
                    p = res["probs"]["hybrid"][k]
                    o = eng.market["1x2_odds"][k]
                    ev = (p*o-1)*100
                    kel = calc_risk_adj_kelly(ev, p*(o-1)**2 - (ev/100)**2, risk_scale, p)
                    r1x2.append({"選項":t, "賠率":o, "EV":f"{ev:.1f}%", "Kelly":f"{kel:.1f}%"})
                    if ev > 0.5: best_bets.append({"sel":t, "odds":o, "stake": unit_stake*kel/100})
                st.dataframe(pd.DataFrame(r1x2))
                
                # AH Table
                rah = []
                for hcap in [-0.5, 0.5]:
                    ev = eng.ah_ev(M, hcap, 1.9)
                    rah.append({"盤口":hcap, "EV":f"{ev:.1f}%"})
                st.dataframe(pd.DataFrame(rah))
                
                # Add to Cart
                if best_bets:
                    s_bet = st.selectbox("加入購物車", [f"{b['sel']} @ {b['odds']}" for b in best_bets])
                    if st.button("➕"):
                        sel = next(b for b in best_bets if f"{b['sel']} @ {b['odds']}" == s_bet)
                        st.session_state.cart.append(sel)
                        st.success("已加入")
                        st.rerun()

            with t_vis:
                if HAS_PLOTLY:
                    st.plotly_chart(plot_radar_chart(res['lh'], res['la']))
                    st.plotly_chart(plot_score_heatmap(M))
                else: st.warning("No Plotly")

            with t_sand:
                st.subheader("🔮 沙盤推演")
                mod = st.slider("主隊進攻調整", 0.5, 1.5, 1.0)
                nlh = res['lh'] * mod
                st.metric("新主勝預期", f"{nlh:.2f}")

    elif app_mode == "📈 實戰績效回顧":
        st.title("📈 績效回顧")
        df = ptrader.load_bets()
        if not df.empty:
            edited = st.data_editor(
                df, 
                num_rows="dynamic", 
                key="editor",
                column_config={
                    "Result": st.column_config.SelectboxColumn(
                        "比賽結果",
                        options=["Pending", "Win", "Lose", "Void"],
                        required=True
                    )
                }
            )
            if st.button("💾 更新損益"):
                ptrader.save_bets(edited) 
                st.success("Updated")
                st.rerun()
            if HAS_PLOTLY and "pnl" in df.columns:
                df["CumPnL"] = df["pnl"].cumsum()
                st.plotly_chart(px.line(df, x="date", y="CumPnL"))
        else: st.info("無數據")

    elif app_mode == "🔧 參數校正實驗室":
        st.header("🔧 參數校正")
        files = st.file_uploader("CSV", accept_multiple_files=True)
        if files:
            dfs = [preprocess_uploaded_data(pd.read_csv(f)) for f in files]
            full = pd.concat([d for d in dfs if not d.empty])
            if st.button("⚡ MLE"):
                r = fit_params_mle(full)
                if r["success"]: st.success(f"Lam3={r['lam3']:.2f}")

    elif app_mode == "🛡️ 風險對沖實驗室":
        st.title("🛡️ 風險對沖實驗室")
        tab_arb, tab_lay, tab_port = st.tabs(["⚡ 1x2 套利", "📉 交易所對沖", "📊 智能組合優化"])
        
        with tab_arb:
            st.subheader("無風險套利計算 (Arbitrage)")
            c1, c2, c3 = st.columns(3)
            o1 = c1.number_input("主勝賠率", 2.0); o2 = c2.number_input("和局賠率", 3.0); o3 = c3.number_input("客勝賠率", 4.0)
            inv = 1/o1+1/o2+1/o3
            if inv<1: st.success(f"發現套利機會! ROI: {1/inv-1:.1%}")
            else: st.info(f"無套利空間 (Book: {inv:.2%})")

        with tab_lay:
            st.subheader("交易所對沖計算器 (Back-Lay)")
            c1, c2 = st.columns(2)
            b_o = c1.number_input("Back 賠率", 1.01, 10.0, 2.5)
            stake = c1.number_input("Back 本金", 10, 1000, 100)
            l_o = c2.number_input("Lay 賠率", 1.01, 10.0, 2.6)
            comm = c2.number_input("佣金 %", 0.0, 5.0, 2.0)/100
            if l_o>1:
                lay_s = (stake*b_o)/(l_o-comm)
                st.metric("建議 Lay 金額", f"${lay_s:.2f}")

        with tab_port:
            st.subheader("智能組合優化 (Portfolio Optimization)")
            if st.session_state.get("res"):
                res = st.session_state.res
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
        
    elif app_mode == "📚 劇本查詢":
        st.dataframe(pd.DataFrame([{"N":v["name"]} for k,v in RegimeMemory().history_db.items()]))

    # [V42.0] New Admin Panel
    elif app_mode == "👑 用戶管理 (Admin)":
        if not st.session_state.is_admin:
            st.error("您沒有權限訪問此頁面")
        else:
            st.header("👑 用戶管理後台")
            st.info("這裡是最高權限區，請謹慎操作")
            
            users_df = auth.get_all_users()
            st.dataframe(users_df, use_container_width=True)
            
            with st.expander("🗑️ 刪除用戶"):
                del_user = st.selectbox("選擇要刪除的用戶", users_df['username'])
                if st.button("確認刪除 (含所有注單)", type="primary"):
                    if del_user == 'admin':
                        st.error("無法刪除超級管理員")
                    elif auth.delete_user(del_user):
                        st.success(f"已刪除用戶 {del_user} 及其所有數據")
                        st.rerun()
                    else:
                        st.error("刪除失敗")
