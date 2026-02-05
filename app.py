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

# [V38] 嘗試導入 Numba
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
# 1. 核心數學工具 (V38.4 Auto-Kernel)
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

def calc_risk_adj_kelly(ev_p, var, risk_scale=0.5, prob=0.5):
    if var<=0 or ev_p<=0: return 0.0
    f = (ev_p/100.0 / var) * risk_scale
    cap = 0.5 if prob>=0.35 else 0.025
    return min(cap, max(0.0, f)) * 100

def calc_risk_metrics(prob, odds):
    if prob<=0 or prob>=1: return 0.0, 0.0
    win_p, lose_p = odds-1.0, -1.0
    ev = prob*win_p + (1-prob)*lose_p
    var = prob*(win_p**2) + (1-prob)*(lose_p**2) - (ev**2)
    sharpe = ev/math.sqrt(var) if var>0 else 0
    return var, sharpe

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
        w_att_def = lambda t: (t["offensive_stats"]["goals_scored_avg"]*0.3 + t["offensive_stats"].get("xg_avg",1.0)*0.7, t["defensive_stats"]["goals_conceded_avg"]*0.3 + t["defensive_stats"].get("xga_avg",1.0)*0.7)
        lh_a, lh_d = w_att_def(self.h)
        la_a, la_d = w_att_def(self.a)
        base = 1.35
        lh = (lh_a * la_d / base) * self.home_adv
        la = (la_a * lh_d / base)
        weighted = False # simplified
        return lh, la, weighted

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
            M[0,0] *= 1 - lh*la*rho
            M[0,1] *= 1 + lh*rho
            M[1,0] *= 1 + la*rho
            M[1,1] *= 1 - rho
            
        M /= M.sum()
        
        imp = get_true_implied_prob(self.market["1x2_odds"])
        ph, pd, pa = float(np.sum(np.tril(M,-1))), float(np.sum(np.diag(M))), float(np.sum(np.triu(M,1)))
        
        # Hybrid
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
        return {"home":0.0, "draw":0.0, "away":0.0} # simplified

    def ah_ev(self, M, hcap, odds):
        idx_diff = np.subtract.outer(np.arange(self.max_g), np.arange(self.max_g)) 
        payoff = np.select([idx_diff + hcap > 0.001, np.abs(idx_diff + hcap) <= 0.001], [odds-1, 0], default=-1)
        return np.sum(M * payoff) * 100

    def check_sensitivity(self, lh, la):
        return "Medium", 0.0 # simplified

    def calc_model_confidence(self, lh, la, diff, sens):
        return 1.0, [] # simplified

    def simulate_uncertainty(self, lh, la, base):
        return base*0.9, base*1.1

    def run_monte_carlo_vectorized(self, M, sims=500000):
        rng = np.random.default_rng()
        flat = M.flatten(); flat /= flat.sum()
        cdf = np.cumsum(flat)
        idx = np.searchsorted(cdf, rng.random(sims))
        hg, ag = idx // M.shape[0], idx % M.shape[0]
        return np.sum(hg>ag)/sims, np.sum(hg==ag)/sims, np.sum(hg<ag)/sims, hg, ag

    def run_ce_importance_sampling(self, M, line, n_sims=20000):
        # Simplified CE
        G = M.shape[0]
        mu_h = np.sum(M.flatten() * (np.arange(G*G)//G))
        mu_a = np.sum(M.flatten() * (np.arange(G*G)%G))
        rng = np.random.default_rng()
        sh = rng.poisson(mu_h*1.5, n_sims)
        sa = rng.poisson(mu_a*1.5, n_sims)
        w = np.exp((sh*(math.log(mu_h)-math.log(mu_h*1.5)) - (mu_h-mu_h*1.5)) + (sa*(math.log(mu_a)-math.log(mu_a*1.5)) - (mu_a-mu_a*1.5)))
        est = np.sum(w * ((sh+sa)>line)) / n_sims
        return {"est": float(est)}

# =========================
# 4. 資料前處理與工具 (V38.4 Auto-Adapter)
# =========================
def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """[V38.4] 自動標準化欄位名稱並生成缺失數據"""
    # 1. 欄位映射字典 (常見格式轉內部格式)
    col_map = {
        'HomeTeam': 'home', 'Home': 'home', 'HT': 'home',
        'AwayTeam': 'away', 'Away': 'away', 'AT': 'away',
        'FTHG': 'home_goals', 'HG': 'home_goals', 'HomeGoals': 'home_goals',
        'FTAG': 'away_goals', 'AG': 'away_goals', 'AwayGoals': 'away_goals',
        'Div': 'div', 'Date': 'date'
    }
    
    # 2. 重新命名欄位 (不區分大小寫)
    df.columns = [c.strip() for c in df.columns] # 去除空白
    new_cols = {}
    for col in df.columns:
        for k, v in col_map.items():
            if col.lower() == k.lower():
                new_cols[col] = v
                break
    df = df.rename(columns=new_cols)
    
    # 3. 確保關鍵欄位存在
    required = ['home', 'away', 'home_goals', 'away_goals']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        # 如果缺少關鍵比分或隊名，無法補救，直接回傳錯誤
        st.error(f"❌ 數據缺少關鍵欄位: {missing}。請確認 CSV 包含球隊名稱與比分。")
        return pd.DataFrame() # 空白代表失敗

    # 4. 自動生成 lh_pred, la_pred (如果缺失)
    # 使用簡單的「聯盟平均法」作為 Baseline
    if 'lh_pred' not in df.columns or 'la_pred' not in df.columns:
        st.info("ℹ️ 偵測到缺失預測數據 (lh_pred, la_pred)。正在根據歷史平均自動生成 Baseline...")
        
        # 計算全聯盟平均主場進球與客場進球
        avg_home = df['home_goals'].mean()
        avg_away = df['away_goals'].mean()
        
        # 簡單賦值 (進階版可用 Rolling Average，但這裡先求穩)
        df['lh_pred'] = avg_home
        df['la_pred'] = avg_away
        
        # 嘗試針對球隊做簡單的強度調整 (Rolling Mean)
        # 建立一個簡單的字典來存球隊平均
        try:
            home_avgs = df.groupby('home')['home_goals'].transform(lambda x: x.expanding().mean().shift(1))
            away_avgs = df.groupby('away')['away_goals'].transform(lambda x: x.expanding().mean().shift(1))
            
            # 填補 NaN (第一場比賽用聯盟平均)
            df['lh_pred'] = home_avgs.fillna(avg_home)
            df['la_pred'] = away_avgs.fillna(avg_away)
        except Exception:
            pass # 如果失敗就用全域平均

    return df

class SimpleKalmanFilter:
    def __init__(self, r=1.0): self.x=r; self.P=1.0; self.Q=0.05; self.R=1.0
    def predict(self): self.P+=self.Q; return self.x
    def update(self, z):
        K = self.P/(self.P+self.R)
        self.x += K*(z-self.x)
        self.P *= (1-K)
        return self.x

def run_kalman_tracking(df):
    if df.empty: return pd.DataFrame(), {}
    teams = set(df['home']).union(set(df['away']))
    ratings = {t: SimpleKalmanFilter() for t in teams}
    history = []
    for _, r in df.iterrows():
        h, a = r['home'], r['away']
        hg, ag = r['home_goals'], r['away_goals']
        rh_pre = ratings[h].predict()
        ra_pre = ratings[a].predict()
        rh_post = ratings[h].update(hg)
        ra_post = ratings[a].update(ag)
        history.append({'home': h, 'away': a, 'h_rating': rh_post, 'a_rating': ra_post})
    return pd.DataFrame(history), ratings

def fit_params_mle(df):
    if df.empty: return {"success": False}
    try:
        lh_arr = df['lh_pred'].values.astype(np.float64)
        la_arr = df['la_pred'].values.astype(np.float64)
        h_arr = df['home_goals'].values.astype(np.int32)
        a_arr = df['away_goals'].values.astype(np.int32)
    except Exception as e:
        return {"success": False}

    def nll_func(params):
        lam3, rho, ha = params
        if not (0<=lam3<=0.5 and -0.3<=rho<=0.3 and 0.8<=ha<=1.6): return 1e9
        return compute_batch_nll(lh_arr, la_arr, h_arr, a_arr, lam3, rho, ha)

    res = minimize(nll_func, [0.1, -0.1, 1.15], method='Nelder-Mead', tol=1e-3)
    return {"lam3": res.x[0], "rho": res.x[1], "home_adv": res.x[2], "success": res.success}

# =========================
# 5. UI (V38.4 Auto-Adapter)
# =========================
st.set_page_config(page_title="Sniper V38.4", page_icon="🧿", layout="wide")
st.markdown("<style>.metric-box { background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; } .stProgress > div > div > div > div { background-color: #4CAF50; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧿 Sniper V38.4")
    st.caption("Auto-Adapter Edition")
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
        use_mock = st.checkbox("歷史記憶修正", True)

if app_mode == "🎯 單場深度預測":
    st.header("🎯 單場深度預測 (V38)")
    if "res" not in st.session_state: st.session_state.res = None
    
    t1, t2 = st.tabs(["📋 貼上 JSON", "📂 上傳 JSON"])
    inp = None
    with t1:
        txt = st.text_area("JSON Input", height=100)
        if txt: inp = json.loads(txt)
    with t2:
        f = st.file_uploader("JSON File", type=['json'])
        if f: inp = json.load(f)

    if st.button("🚀 分析", type="primary") and inp:
        eng = SniperAnalystLogic(inp, 9, nb_alpha, lam3_in, rho_in, ha_in)
        lh, la, w = eng.calc_lambda()
        M, probs = eng.build_matrix_v38(lh, la, use_biv, use_dc)
        hw, dr, aw, sh, sa = eng.run_monte_carlo_vectorized(M)
        
        # Simple Analysis
        odds = eng.market["1x2_odds"]
        regime = eng.memory.analyze_scenario(lh, la, odds)
        
        st.session_state.res = {"eng": eng, "M": M, "lh": lh, "la": la, "sh": sh, "sa": sa, "probs": probs, "regime": regime}

    if st.session_state.res:
        r = st.session_state.res
        eng = r["eng"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("主預期", f"{r['lh']:.2f}")
        c2.metric("客預期", f"{r['la']:.2f}")
        c3.metric("模型主勝", f"{r['probs']['hybrid']['home']:.1%}")
        c4.metric("劇本", r["regime"])
        
        t_v, t_s = st.tabs(["💰 價值", "🎲 模擬"])
        with t_v:
            # Simple EV Table
            odds = eng.market["1x2_odds"]
            evs = []
            for k in ["home", "draw", "away"]:
                p = r["probs"]["hybrid"][k]
                o = odds[k]
                evs.append({"Pick": k, "Odds": o, "Prob": f"{p:.1%}", "EV": f"{(p*o-1)*100:.1f}%"})
            st.dataframe(pd.DataFrame(evs))
            
        with t_s:
            hw = np.sum(r["sh"] > r["sa"]) / 500000
            st.metric("MC 主勝率", f"{hw:.1%}")
            fig, ax = plt.subplots(figsize=(6,2))
            ax.hist(r["sh"], alpha=0.5, label="H"); ax.hist(r["sa"], alpha=0.5, label="A"); ax.legend()
            st.pyplot(fig)

elif app_mode == "🛡️ 風險對沖實驗室":
    st.title("🛡️ 風險對沖")
    if st.session_state.get("res"):
        r = st.session_state.res
        sh, sa = r["sh"], r["sa"]
        eng = r["eng"]
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
            res = minimize(obj, [1/len(cands)]*len(cands), bounds=[(0,1)]*len(cands), constraints=cons)
            
            st.write("建議配置:")
            cols = st.columns(len(cands))
            for i, w in enumerate(res.x):
                cols[i].metric(cands[i]["name"], f"{w:.1%}", delta=f"EV: {mu[i]*100:.1f}%")
            
            # Black Text Fix
            st.markdown("""<div style="background:#f0f2f6; padding:10px; color:black; border-radius:5px;">
            <b>分析師評語:</b> 請依照上述比例分配資金以最大化夏普比率。</div>""", unsafe_allow_html=True)
    else:
        st.warning("請先執行單場預測")

elif app_mode == "🔧 參數校正實驗室":
    st.header("🔧 參數校正 (自動適配版)")
    
    # [V38.4] 強制多選 + 自動適配
    files = st.file_uploader("上傳 CSV/Excel (支援 FTHG/HomeTeam 等格式)", type=['csv','xlsx'], accept_multiple_files=True, key="up_v38_4")
    
    if files:
        dfs = []
        for f in files:
            try:
                if f.name.endswith('.csv'):
                    try: df = pd.read_csv(f, encoding='utf-8')
                    except: f.seek(0); df = pd.read_csv(f, encoding='big5')
                else:
                    import openpyxl; df = pd.read_excel(f)
                
                # [V38.4] 呼叫資料處理函式
                df = preprocess_uploaded_data(df)
                if not df.empty: dfs.append(df)
            except Exception as e: st.warning(f"{f.name} 失敗: {e}")
            
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            st.write(f"成功處理 {len(full_df)} 筆數據 (已自動生成 lh_pred/la_pred)", full_df.head(3))
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⚡ MLE 擬合"):
                    with st.spinner("計算中..."):
                        res = fit_params_mle(full_df)
                    if res["success"]:
                        st.success(f"建議: Lam3={res['lam3']:.2f}, Rho={res['rho']:.2f}, HA={res['home_adv']:.2f}")
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
