import streamlit as st
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 核心數學工具
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

def calc_kelly(prob, odds, fraction=0.4):
    if prob <= 0 or odds <= 1:
        return 0.0
    b = odds - 1.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * fraction) * 100

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

    # V25 新增：蒙地卡羅模擬核心
    def run_monte_carlo(self, lh, la, sims=5000):
        # 模擬 5000 場比賽的進球分佈
        home_goals = np.random.poisson(lh, sims)
        away_goals = np.random.poisson(la, sims)
        
        results = []
        for hg, ag in zip(home_goals, away_goals):
            if hg > ag: results.append("home")
            elif hg == ag: results.append("draw")
            else: results.append("away")
            
        return home_goals, away_goals, results

# =========================
# 3. Streamlit UI 介面
# =========================
st.set_page_config(page_title="狙擊手分析 V25.0 UI", page_icon="⚽", layout="wide")

st.title("⚽ 狙擊手 V25.0 戰情室")
st.markdown("### 專業足球數據分析：獲利計算 x 戰局模擬 x 價值注單")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    # V25 新增：獲利計算設定
    unit_stake = st.number_input("💰 設定單注本金 ($)", min_value=10, value=100, step=10, help="輸入你的單注金額，系統將自動計算預計獲利")
    st.divider()
    nb_alpha = st.slider("負二項分佈 Alpha (變異數)", 0.05, 0.20, 0.12, 0.01)
    max_g = st.number_input("最大進球數運算範圍", 5, 15, 9)
    kelly_frac = st.slider("凱利公式比例 (Kelly Fraction)", 0.1, 1.0, 0.4, 0.1)

# --- 數據輸入區 ---
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

# --- 執行分析按鈕 ---
if st.button("🚀 開始全方位分析", type="primary"):
    if not input_data:
        st.error("請先輸入有效的比賽數據！")
    else:
        # 初始化分析引擎
        engine = SniperAnalystLogic(input_data, max_g, nb_alpha)
        
        # 1. 計算數據
        lh, la = engine.calc_lambda()
        M = engine.build_ensemble_matrix(lh, la)
        market_bonus = engine.get_market_trend_bonus()
        
        # 2. 顯示對戰資訊
        st.divider()
        col1, col2, col3 = st.columns([1, 0.2, 1])
        with col1:
            st.markdown(f"<h3 style='text-align: right; color: #1f77b4;'>{engine.h['name']}</h3>", unsafe_allow_html=True)
            st.metric("預期進球 (Lambda)", f"{lh:.2f}")
        with col2:
            st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<h3 style='text-align: left; color: #ff7f0e;'>{engine.a['name']}</h3>", unsafe_allow_html=True)
            st.metric("預期進球 (Lambda)", f"{la:.2f}")

        # 3. 計算機率
        prob_h = float(np.sum(np.tril(M,-1)))
        prob_d = float(np.sum(np.diag(M)))
        prob_a = float(np.sum(np.triu(M,1)))

        # --- V25 Tab 分頁架構 ---
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📊 投注價值與獲利", "🧠 智能裁決", "🎯 波膽分佈", "🎲 戰局模擬與雷達"])

        candidates = []

        # --- Tab 1: 價值分析 (含獲利計算) ---
        with res_tab1:
            st.subheader("💰 獨贏 (1x2) 分析")
            data_1x2 = []
            for tag, prob, key in [("主勝", prob_h, "home"), ("和局", prob_d, "draw"), ("客勝", prob_a, "away")]:
                odd = engine.market["1x2_odds"][key]
                ev = (prob * odd - 1) * 100 + market_bonus[key]
                # V25: 獲利計算
                profit = (odd - 1) * unit_stake
                data_1x2.append([tag, f"{prob*100:.1f}%", odd, f"{ev:+.1f}%", f"${profit:.1f}"])
                if ev > 1.5:
                    candidates.append({"type":"1x2", "pick":tag, "ev":ev, "odds":odd, "prob":prob})
            
            st.table(pd.DataFrame(data_1x2, columns=["選項", "模型機率", "賠率", "EV", "預計獲利"]))

            col_ah, col_ou = st.columns(2)
            with col_ah:
                st.subheader("🛡️ 亞盤 (Handicap)")
                data_ah = []
                for hcap in engine.market["handicaps"]:
                    ev = engine.ah_ev(M, hcap, engine.market["target_odds"]) + market_bonus["home"]
                    profit = (engine.market["target_odds"] - 1) * unit_stake
                    data_ah.append([f"主 {hcap:+}", f"{ev:+.1f}%", f"${profit:.1f}"])
                    if ev > 2:
                        candidates.append({"type":"AH", "pick":f"主 {hcap:+}", "ev":ev, "odds":engine.market["target_odds"], "prob":0.5+ev/200})
                st.table(pd.DataFrame(data_ah, columns=["盤口", "EV", "預計獲利"]))

            with col_ou:
                st.subheader("📐 大小球 (Over/Under)")
                data_ou = []
                for line in engine.market["goal_lines"]:
                    o_prob = sum(M[i,j] for i in range(9) for j in range(9) if i+j>line)
                    ev_o = (o_prob * engine.market["target_odds"] - 1) * 100
                    profit = (engine.market["target_odds"] - 1) * unit_stake
                    data_ou.append([f"大 {line}", f"{o_prob*100:.1f}%", f"{ev_o:+.1f}%", f"${profit:.1f}"])
                    if ev_o > 2:
                        candidates.append({"type":"OU", "pick":f"大 {line}", "ev":ev_o, "odds":engine.market["target_odds"], "prob":o_prob})
                st.table(pd.DataFrame(data_ou, columns=["盤口", "機率", "EV", "預計獲利"]))

            st.subheader("📝 最佳投資組合 (Top Picks)")
            if candidates:
                final_list = sorted(candidates, key=lambda x:x["ev"], reverse=True)
                reco_data = []
                for p in final_list[:3]:
                    kelly = calc_kelly(p["prob"], p["odds"], kelly_frac)
                    profit = (p['odds'] - 1) * unit_stake
                    reco_data.append([f"[{p['type']}] {p['pick']}", p['odds'], f"{p['ev']:+.1f}%", f"{kelly:.1f}%", f"${profit:.1f}"])
                st.dataframe(pd.DataFrame(reco_data, columns=["選項", "賠率", "EV", "建議注碼%", "預計獲利"]), use_container_width=True)
                st.caption(f"* 預計獲利基於本金 ${unit_stake} 計算")
            else:
                st.info("無高 EV 選項推薦")

        # --- Tab 2: 智能裁決 ---
        with res_tab2:
            st.subheader("🧠 模型裁決與警報")
            total_xg = lh + la
            if total_xg > 3.5: st.warning(f"🟠 高變異節奏 (Total xG: {total_xg:.2f}) - 攻防轉換快，紅牌點球影響大。")
            elif total_xg > 2.5: st.success(f"🟢 中性節奏 (Total xG: {total_xg:.2f}) - 模型穩定性佳。")
            else: st.info(f"🔵 低節奏 (Total xG: {total_xg:.2f}) - 爆冷多來自定位球。")

            if candidates:
                top_pick = sorted(candidates, key=lambda x:x["ev"], reverse=True)[0]
                market_implied = 1.0 / top_pick['odds']
                model_prob = top_pick['prob']
                edge_diff = model_prob - market_implied
                
                st.markdown("---")
                st.write(f"**最佳選項 [{top_pick['pick']}] 深度檢核：**")
                c1, c2 = st.columns(2)
                c1.metric("模型機率", f"{model_prob*100:.1f}%")
                c2.metric("市場隱含", f"{market_implied*100:.1f}%")

                if edge_diff < 0: st.error("🔴 虛高風險 (High Odds Trap)：EV 來自高賠率槓桿，實際勝率低。建議減半注碼。")
                elif edge_diff < 0.03: st.warning("🟠 邊際優勢 (Thin Edge)：優勢不明顯，嚴格遵守注碼，不追單。")
                else: st.success("🟢 真實價值 (True Value)：發現顯著機率偏差，信心買入。")
            
            if len(candidates) >= 2:
                final_list = sorted(candidates, key=lambda x:x["ev"], reverse=True)
                p1 = final_list[0]; p2 = final_list[1]
                def get_dir(n):
                    if "主" in n: return "HOME"
                    if "客" in n: return "AWAY"
                    if "大" in n: return "OVER"
                    return "NONE"
                if get_dir(p1['pick']) != "NONE" and get_dir(p1['pick']) == get_dir(p2['pick']):
                    st.error(f"⚠️ 資金控管警報：Top 1 與 Top 2 方向重疊！建議分攤注碼。")

        # --- Tab 3: 波膽 ---
        with res_tab3:
            st.subheader("🎯 波膽 (Correct Score) 熱力圖")
            df_cs = pd.DataFrame(M[:6, :6], columns=[f"客 {j}" for j in range(6)], index=[f"主 {i}" for i in range(6)])
            st.dataframe(df_cs.style.format("{:.1%}", subset=None).background_gradient(cmap="Blues", axis=None))
            st.write("**高價值波膽推薦：**")
            for s, odd in engine.market["cs_odds"].items():
                try:
                    i, j = map(int, s.split(":"))
                    prob = M[i, j]
                    ev = (prob * odd - 1) * 100
                    if ev > 10:
                        profit = (odd - 1) * unit_stake
                        st.write(f"- **{s}** @ {odd} (機率 {prob*100:.1f}%, EV {ev:+.1f}%) -> 獲利: ${profit:.1f}")
                except: pass

        # --- V25 Tab 4: 模擬與雷達 ---
        with res_tab4:
            st.subheader("🎲 蒙地卡羅模擬 (5,000 場預演)")
            sim_h, sim_a, sim_res = engine.run_monte_carlo(lh, la)
            
            wh = sim_res.count("home") / 5000
            wd = sim_res.count("draw") / 5000
            wa = sim_res.count("away") / 5000
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("模擬主勝率", f"{wh:.1%}")
            sc2.metric("模擬和局率", f"{wd:.1%}")
            sc3.metric("模擬客勝率", f"{wa:.1%}")
            
            st.write("**進球數機率分佈 (Histogram)**")
            fig, ax = plt.subplots(figsize=(10, 4))
            counts_h, bins_h = np.histogram(sim_h, bins=range(10), density=True)
            ax.bar(bins_h[:-1]-0.15, counts_h, width=0.3, color='#1f77b4', alpha=0.7, label='Home Goals')
            counts_a, bins_a = np.histogram(sim_a, bins=range(10), density=True)
            ax.bar(bins_a[:-1]+0.15, counts_a, width=0.3, color='#ff7f0e', alpha=0.7, label='Away Goals')
            ax.set_xticks(range(9))
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            st.info("💡 藍柱: 主隊進球機率 | 橘柱: 客隊進球機率。重疊越高代表平局或小球機率越大。")

            st.divider()
            st.subheader("⚔️ 綜合戰力雷達圖")
            categories = ['Attack', 'Defense', 'Form', 'Home/Away', 'Motivation']
            
            def get_score(stats, is_home):
                att = min(10, stats["offensive_stats"]["xg_avg"] * 4)
                deff = min(10, (3 - stats["defensive_stats"]["xga_avg"]) * 3.5)
                form = sum(stats["context_modifiers"]["recent_form_trend"]) * 2
                adv = stats["general_strength"]["home_advantage_weight"] * 5
                motiv = 8 if stats["context_modifiers"]["motivation"] != "normal" else 5
                return [att, deff, form, adv, motiv]

            h_scores = get_score(engine.h, True)
            a_scores = get_score(engine.a, False)
            
            N = len(categories)
            angles = [n / float(N) * 2 * math.pi for n in range(N)]
            angles += angles[:1]
            h_scores += h_scores[:1]
            a_scores += a_scores[:1]
            
            fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax_radar.plot(angles, h_scores, linewidth=2, linestyle='solid', label='Home', color='#1f77b4')
            ax_radar.fill(angles, h_scores, '#1f77b4', alpha=0.2)
            ax_radar.plot(angles, a_scores, linewidth=2, linestyle='solid', label='Away', color='#ff7f0e')
            ax_radar.fill(angles, a_scores, '#ff7f0e', alpha=0.2)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            st.pyplot(fig_radar)
