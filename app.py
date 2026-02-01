import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 核心數學工具 (V26.2 升級版)
# =========================
def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def nb_pmf(k, mu, alpha):
    if alpha <= 0:
        return poisson_pmf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    coeff = math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1))
    return float(coeff * (p ** r) * ((1 - p) ** k))

# V26.2 新增：風險調整注碼計算 (Risk-Adjusted Kelly)
def calc_risk_adj_kelly(ev_percent, variance, risk_scale=0.5):
    """
    使用 Mean-Variance 優化邏輯替代傳統 Kelly
    f = (EV / Variance) * Risk_Scale
    ev_percent: 期望值 (例如 5.0 代表 5%)
    variance: 收益的變異數
    risk_scale: 風險縮放係數 (類似 Kelly Fraction)
    """
    if variance <= 0 or ev_percent <= 0:
        return 0.0
    
    # 將百分比轉回小數計算
    ev = ev_percent / 100.0
    
    # 公式：f = E / Var * c
    f = (ev / variance) * risk_scale
    
    # 限制最大單注不超過 50% (安全閥)
    return min(0.5, max(0.0, f)) * 100

# =========================
# 2. 分析引擎邏輯
# =========================
class SniperAnalystLogic:
    def __init__(self, json_data, max_g=9, nb_alpha=0.12):
        self.data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        self.h = self.data["home"]
        self.a = self.data["away"]
        self.market = self.data["market_data"]
        self.max_g = max_g
        self.nb_alpha = nb_alpha

    def calc_lambda(self):
        league_base = 1.35
        def att_def(team):
            att = 0.4*team["offensive_stats"]["goals_scored_avg"] + 0.6*team["offensive_stats"]["xg_avg"]
            deff = 0.4*team["defensive_stats"]["goals_conceded_avg"] + 0.6*team["defensive_stats"]["xga_avg"]
            return att, deff

        h_att, h_def = att_def(self.h)
        a_att, a_def = att_def(self.a)

        if self.h["context_modifiers"]["missing_key_defender"]: h_def *= 1.20
        if self.a["context_modifiers"]["missing_key_defender"]: a_def *= 1.15

        h_adv = self.h["general_strength"]["home_advantage_weight"]
        lh = (h_att * a_def / league_base) * h_adv
        la = (a_att * h_def / league_base)

        if self.h["context_modifiers"]["motivation"] == "survival": lh *= 1.05
        if self.a["context_modifiers"]["motivation"] == "title_race": la *= 1.05

        return lh, la

    def get_market_trend_bonus(self):
        bonus = {"home":0.0,"draw":0.0,"away":0.0}
        op = self.market.get("opening_odds")
        cu = self.market.get("1x2_odds")
        if not op or not cu: return bonus
        for k in bonus:
            drop = max(0.0,(op[k]-cu[k])/op[k])
            bonus[k] = min(3.0, drop*30.0)
        return bonus

    def build_ensemble_matrix(self, lh, la):
        G = self.max_g
        Mp = np.zeros((G,G))
        Mn = np.zeros((G,G))
        for i in range(G):
            for j in range(G):
                Mp[i,j] = poisson_pmf(i,lh)*poisson_pmf(j,la)
                Mn[i,j] = nb_pmf(i,lh,self.nb_alpha)*nb_pmf(j,la,self.nb_alpha)
        M = 0.6*Mp + 0.4*Mn
        rho = -0.18 if self.h["style_of_play"]["volatility"]=="high" else -0.13
        for (i,j),f in {(0,0):1-lh*la*rho,(1,0):1+la*rho,(0,1):1+lh*rho,(1,1):1-rho}.items():
            if i<G and j<G: M[i,j] *= f
        return M/M.sum()

    def ah_ev(self, M, hcap, odds):
        ev = 0.0
        for i in range(self.max_g):
            for j in range(self.max_g):
                r = (i-j)+hcap
                if r>0.25: p=odds-1
                elif abs(r-0.25)<1e-9: p=(odds-1)*0.5
                elif abs(r)<1e-9: p=0
                elif abs(r+0.25)<1e-9: p=-0.5
                else: p=-1
                ev += M[i,j]*p
        return ev*100

    def run_monte_carlo(self, lh, la, sims=5000):
        home_goals = np.random.poisson(lh, sims)
        away_goals = np.random.poisson(la, sims)
        results = []
        for hg, ag in zip(home_goals, away_goals):
            if hg > ag: results.append("home")
            elif hg == ag: results.append("draw")
            else: results.append("away")
        return home_goals, away_goals, results

    def check_sensitivity(self, lh, la, pick_type, original_ev):
        M_stress = self.build_ensemble_matrix(lh, la + 0.3)
        prob_h_orig = float(np.sum(np.tril(self.build_ensemble_matrix(lh, la),-1)))
        prob_h_new = float(np.sum(np.tril(M_stress,-1)))
        drop_rate = (prob_h_orig - prob_h_new) / prob_h_orig if prob_h_orig > 0 else 0
        if drop_rate > 0.15: return "High", "脆弱"
        elif drop_rate > 0.08: return "Medium", "普通"
        else: return "Low", "堅固"

    # V26.2 新增：計算變異數 (Variance) 與夏普值 (Sharpe)
    def calc_risk_metrics(self, prob, odds):
        """
        計算該注單的統計風險特徵
        Returns: Variance, Sharpe
        """
        if prob <= 0 or prob >= 1: return 0.0, 0.0
        
        # 贏的時候賺 (odds-1)，輸的時候賠 1
        win_payoff = odds - 1.0
        lose_payoff = -1.0
        
        # E[X] = p * win + (1-p) * lose
        expected_val = prob * win_payoff + (1 - prob) * lose_payoff
        
        # E[X^2] = p * win^2 + (1-p) * lose^2
        expected_sq = prob * (win_payoff**2) + (1 - prob) * (lose_payoff**2)
        
        # Var(X) = E[X^2] - (E[X])^2
        variance = expected_sq - (expected_val**2)
        
        # Sharpe = E[X] / StdDev(X)
        std_dev = math.sqrt(variance)
        sharpe = expected_val / std_dev if std_dev > 0 else 0
        
        return variance, sharpe

# =========================
# 3. Streamlit UI 介面
# =========================
st.set_page_config(page_title="狙擊手分析 V26.2 UI", page_icon="⚽", layout="wide")

st.title("⚽ 狙擊手 V26.2 量化決策版")
st.markdown("### 專業足球數據分析：風險定價 x 波動調節 x 智能注碼")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    unit_stake = st.number_input("💰 設定單注本金 ($)", min_value=10, value=100, step=10)
    st.divider()
    nb_alpha = st.slider("Alpha (變異數)", 0.05, 0.20, 0.12, 0.01)
    max_g = st.number_input("運算範圍", 5, 15, 9)
    # V26.2 更新：改名為風險係數
    risk_scale = st.slider("風險縮放係數 (Risk Scale)", 0.1, 1.0, 0.3, 0.1, help="數值越小，系統面對高波動注單時砍注越兇。建議 0.3-0.5。")

# --- 輸入區 ---
st.info("請選擇數據輸入方式：")
tab1, tab2 = st.tabs(["📋 貼上 JSON 代碼", "📂 上傳 JSON 檔案"])
input_data = None
default_json = """{ "meta_info": { "league_name": "範例聯賽", "match_date": "2026-01-01" }, "market_data": { "handicaps": [0.5, 0.75], "goal_lines": [2.5, 3.0], "target_odds": 1.90, "1x2_odds": { "home": 2.40, "draw": 3.30, "away": 2.50 }, "opening_odds": { "home": 2.30, "draw": 3.30, "away": 2.60 }, "cs_odds": { "1:0": 8.0, "0:1": 8.5, "1:1": 6.5 } }, "home": { "name": "主隊範例", "general_strength": { "home_advantage_weight": 1.15 }, "offensive_stats": { "goals_scored_avg": 1.5, "xg_avg": 1.4 }, "defensive_stats": { "goals_conceded_avg": 1.2, "xga_avg": 1.3 }, "style_of_play": { "volatility": "normal" }, "context_modifiers": { "motivation": "normal", "missing_key_defender": false } }, "away": { "name": "客隊範例", "general_strength": { "home_advantage_weight": 0.9 }, "offensive_stats": { "goals_scored_avg": 1.1, "xg_avg": 1.2 }, "defensive_stats": { "goals_conceded_avg": 1.6, "xga_avg": 1.5 }, "style_of_play": { "volatility": "high" }, "context_modifiers": { "motivation": "normal", "missing_key_defender": true } } }"""

with tab1:
    json_text = st.text_area("在此貼上 JSON", value=default_json, height=150)
    if json_text:
        try: input_data = json.loads(json_text)
        except: st.error("JSON 格式錯誤")
with tab2:
    uploaded_file = st.file_uploader("選擇 .json 或 .txt 檔案", type=['json', 'txt'])
    if uploaded_file:
        try: input_data = json.load(uploaded_file)
        except: st.error("檔案讀取失敗")

# --- 執行分析 ---
if st.button("🚀 開始全方位分析", type="primary"):
    if not input_data:
        st.error("請先輸入有效的比賽數據！")
    else:
        engine = SniperAnalystLogic(input_data, max_g, nb_alpha)
        
        # 1. 基礎計算
        lh, la = engine.calc_lambda()
        M = engine.build_ensemble_matrix(lh, la)
        market_bonus = engine.get_market_trend_bonus()
        
        # 2. 顯示對戰
        st.divider()
        col1, col2, col3 = st.columns([1, 0.2, 1])
        with col1:
            st.markdown(f"<h3 style='text-align: right; color: #1f77b4;'>{engine.h['name']}</h3>", unsafe_allow_html=True)
            st.metric("預期進球", f"{lh:.2f}")
        with col2: st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<h3 style='text-align: left; color: #ff7f0e;'>{engine.a['name']}</h3>", unsafe_allow_html=True)
            st.metric("預期進球", f"{la:.2f}")

        prob_h = float(np.sum(np.tril(M,-1)))
        prob_d = float(np.sum(np.diag(M)))
        prob_a = float(np.sum(np.triu(M,1)))

        # V26.2 Tab 架構
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📊 價值與風險定價", "🧠 智能裁決", "🎯 波膽分佈", "🎲 模擬與雷達"])

        candidates = []

        # --- Tab 1: 價值與風險定價 ---
        with res_tab1:
            st.subheader("💰 獨贏 (1x2) 風險定價")
            rows_1x2 = []
            for tag, prob, key in [("主勝", prob_h, "home"), ("和局", prob_d, "draw"), ("客勝", prob_a, "away")]:
                odd = engine.market["1x2_odds"][key]
                total_ev = (prob * odd - 1) * 100 + market_bonus[key]
                
                # V26.2: 計算 Variance 與 Sharpe
                var, sharpe = engine.calc_risk_metrics(prob, odd)
                
                # V26.2: 計算風險調整注碼
                kelly_pct = calc_risk_adj_kelly(total_ev, var, risk_scale)
                final_stake = unit_stake * (kelly_pct / 100 * 10) # 假設 unit_stake 是基準，這裡只顯示比例概念，或直接顯示建議金額
                final_stake_amt = (unit_stake * kelly_pct) / 10 # 這裡做個調整，讓顯示數字合理化
                
                # 獲利計算 (基於風險調整後的建議注碼，這裡為了簡單顯示，還是用本金算，但在 Top Picks 會用風險注碼)
                profit = (odd - 1) * unit_stake
                
                # 壓力測試
                sens_level, sens_desc = engine.check_sensitivity(lh, la, tag, total_ev)
                
                rows_1x2.append({
                    "選項": tag, "賠率": odd, "EV": f"{total_ev:+.1f}%",
                    "波動 (Var)": f"{var:.2f}",
                    "夏普值": f"{sharpe:.2f}", # 越高越好
                    "壓力": sens_desc, 
                    "建議注碼%": f"{kelly_pct:.1f}%"
                })
                if total_ev > 1.5: 
                    candidates.append({
                        "type":"1x2", "pick":tag, "ev":total_ev, "odds":odd, "prob":prob, 
                        "sens": sens_level, "var": var, "sharpe": sharpe, "kelly": kelly_pct
                    })
            st.dataframe(pd.DataFrame(rows_1x2), use_container_width=True)

            # 亞盤與大小球
            c_ah, c_ou = st.columns(2)
            with c_ah:
                st.subheader("🛡️ 亞盤")
                d_ah = []
                for hcap in engine.market["handicaps"]:
                    ev = engine.ah_ev(M, hcap, engine.market["target_odds"]) + market_bonus["home"]
                    # 亞盤機率近似反推
                    target_o = engine.market["target_odds"]
                    prob_approx = (ev/100.0 + 1) / target_o
                    
                    var, sharpe = engine.calc_risk_metrics(prob_approx, target_o)
                    kelly_pct = calc_risk_adj_kelly(ev, var, risk_scale)
                    sens_level, sens_desc = engine.check_sensitivity(lh, la, "AH", ev)
                    
                    d_ah.append({
                        "盤口": f"主 {hcap:+}", "EV": f"{ev:+.1f}%", 
                        "夏普值": f"{sharpe:.2f}",
                        "建議注碼%": f"{kelly_pct:.1f}%"
                    })
                    if ev > 2: 
                        candidates.append({
                            "type":"AH", "pick":f"主 {hcap:+}", "ev":ev, "odds":target_o, "prob":prob_approx, 
                            "sens":"Medium", "var": var, "sharpe": sharpe, "kelly": kelly_pct
                        })
                st.dataframe(pd.DataFrame(d_ah), use_container_width=True)
            
            with c_ou:
                st.subheader("📐 大小球")
                d_ou = []
                for line in engine.market["goal_lines"]:
                    op = sum(M[i,j] for i in range(9) for j in range(9) if i+j>line)
                    ev = (op * engine.market["target_odds"] - 1) * 100
                    target_o = engine.market["target_odds"]
                    
                    var, sharpe = engine.calc_risk_metrics(op, target_o)
                    kelly_pct = calc_risk_adj_kelly(ev, var, risk_scale)
                    sens_level, sens_desc = engine.check_sensitivity(lh, la, "OU", ev)
                    
                    d_ou.append({
                        "盤口": f"大 {line}", "EV": f"{ev:+.1f}%",
                        "夏普值": f"{sharpe:.2f}",
                        "建議注碼%": f"{kelly_pct:.1f}%"
                    })
                    if ev > 2: 
                        candidates.append({
                            "type":"OU", "pick":f"大 {line}", "ev":ev, "odds":target_o, "prob":op, 
                            "sens":"Medium", "var": var, "sharpe": sharpe, "kelly": kelly_pct
                        })
                st.dataframe(pd.DataFrame(d_ou), use_container_width=True)

            # 最佳推薦 (V26.2: 顯示風險定價後的結果)
            st.subheader("📝 智能投資組合 (Smart Portfolio)")
            if candidates:
                # 排序改用 Sharpe 值 (CP值) 或 EV 綜合考量，這裡我們還是先看 EV，但在表中強調風險
                final = sorted(candidates, key=lambda x:x["ev"], reverse=True)[:3]
                
                no_bet_flag = False
                no_bet_reason = []
                top = final[0]
                if top['sens'] == "High" and top['ev'] < 15:
                    no_bet_flag = True; no_bet_reason.append("首選注單脆弱")
                
                if no_bet_flag:
                    st.error(f"🛑 系統建議觀望 (NO BET)")
                    for r in no_bet_reason: st.write(f"- {r}")
                else:
                    reco = []
                    for p in final:
                        # 這裡計算真實建議下注金額
                        # 假設 unit_stake 是你的 "Max Stake" (例如 100鎂)，kelly_pct 是比例
                        # 為了安全，我們把 kelly_pct 映射到本金
                        bet_amount = unit_stake * (p['kelly'] / 10.0) # 簡單縮放，避免數字太大
                        
                        risk_icon = "🟢" if p['sharpe'] > 0.1 else ("🟡" if p['sharpe'] > 0.05 else "🔴")
                        reco.append([
                            f"[{p['type']}] {p['pick']}", 
                            p['odds'], 
                            f"{p['ev']:+.1f}%", 
                            f"{risk_icon} {p['sharpe']:.3f}", 
                            f"{p['kelly']:.1f}%", 
                            f"${bet_amount:.1f}"
                        ])
                    st.dataframe(pd.DataFrame(reco, columns=["選項", "賠率", "EV", "夏普值(穩)", "風險注碼%", "建議金額"]), use_container_width=True)
                    st.caption("💡 夏普值 (Sharpe) 越高代表風險回報比越好。建議金額已根據波動率自動調整。")
            else:
                st.info("無適合注單")

        # --- Tab 2, 3, 4 維持不變 ---
        with res_tab2:
            st.subheader("🧠 模型裁決")
            total_xg = lh + la
            if total_xg > 3.5: st.warning(f"🟠 高變異節奏 (xG {total_xg:.2f})")
            elif total_xg > 2.5: st.success(f"🟢 中性節奏 (xG {total_xg:.2f})")
            else: st.info(f"🔵 低節奏 (xG {total_xg:.2f})")
            
            if candidates:
                top = sorted(candidates, key=lambda x:x["ev"], reverse=True)[0]
                imp = 1.0/top['odds']
                diff = top['prob'] - imp
                col_c1, col_c2 = st.columns(2)
                col_c1.metric("模型機率", f"{top['prob']*100:.1f}%")
                col_c2.metric("市場隱含", f"{imp*100:.1f}%")
                if diff < 0: st.error("🔴 虛高風險：EV 來自賠率槓桿")
                elif diff < 0.03: st.warning("🟠 邊際優勢：優勢不明顯")
                else: st.success("🟢 真實價值：顯著機率偏差")

        with res_tab3:
            st.subheader("🎯 波膽分佈")
            df_cs = pd.DataFrame(M[:6,:6], columns=[f"客{j}" for j in range(6)], index=[f"主{i}" for i in range(6)])
            st.dataframe(df_cs.style.format("{:.1%}", subset=None).background_gradient(cmap="Blues", axis=None))

        with res_tab4:
            st.subheader("🎲 戰局模擬")
            sh, sa, sr = engine.run_monte_carlo(lh, la)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("主勝率", f"{sr.count('home')/50:.1f}%")
            sc2.metric("和局率", f"{sr.count('draw')/50:.1f}%")
            sc3.metric("客勝率", f"{sr.count('away')/50:.1f}%")
            
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
                return [min(10, stats["offensive_stats"]["xg_avg"]*4), min(10, (3-stats["defensive_stats"]["xga_avg"])*3.5), sum(stats["context_modifiers"]["recent_form_trend"])*2, stats["general_strength"]["home_advantage_weight"]*5, 8 if stats["context_modifiers"]["motivation"]!="normal" else 5]
            
            hs, ans = get_s(engine.h), get_s(engine.a)
            N = len(cats); ang = [n/float(N)*2*math.pi for n in range(N)]; ang+=ang[:1]; hs+=hs[:1]; ans+=ans[:1]
            figr, axr = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            axr.plot(ang, hs, color='#1f77b4', label='Home'); axr.fill(ang, hs, '#1f77b4', alpha=0.2)
            axr.plot(ang, ans, color='#ff7f0e', label='Away'); axr.fill(ang, ans, '#ff7f0e', alpha=0.2)
            axr.set_xticks(ang[:-1]); axr.set_xticklabels(cats); axr.legend()
            st.pyplot(figr)
