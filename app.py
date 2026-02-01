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

# =========================
# 3. Streamlit UI 介面
# =========================
st.set_page_config(page_title="狙擊手分析 V24.1 UI", page_icon="⚽", layout="wide")

st.title("⚽ 狙擊手 V24.1 分析系統")
st.markdown("### 專業足球數據分析與價值注單計算")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    # 新增本金設定
    unit_stake = st.number_input("💰 設定單注本金 ($)", min_value=10, value=100, step=10, help="輸入你的單注金額，系統將自動計算預計獲利")
    st.divider()
    nb_alpha = st.slider("負二項分佈 Alpha (變異數)", 0.05, 0.20, 0.12, 0.01)
    max_g = st.number_input("最大進球數運算範圍", 5, 15, 9)
    kelly_frac = st.slider("凱利公式比例 (Kelly Fraction)", 0.1, 1.0, 0.4, 0.1)

# --- 數據輸入區 ---
st.info("請選擇數據輸入方式：")
tab1, tab2 = st.tabs(["📋 貼上 JSON 代碼", "📂 上傳 JSON 檔案"])

input_data = None

# 預設範本
default_json = """
{
  "meta_info": { "league_name": "範例聯賽", "match_date": "2026-01-01" },
  "market_data": {
    "handicaps": [0.5, 0.75], "goal_lines": [2.5, 3.0], "target_odds": 1.90,
    "1x2_odds": { "home": 2.40, "draw": 3.30, "away": 2.50 },
    "opening_odds": { "home": 2.30, "draw": 3.30, "away": 2.60 },
    "cs_odds": { "1:0": 8.0, "0:1": 8.5, "1:1": 6.5 }
  },
  "home": {
    "name": "主隊範例", "general_strength": { "home_advantage_weight": 1.15 },
    "offensive_stats": { "goals_scored_avg": 1.5, "xg_avg": 1.4 },
    "defensive_stats": { "goals_conceded_avg": 1.2, "xga_avg": 1.3 },
    "style_of_play": { "volatility": "normal" },
    "context_modifiers": { "motivation": "normal", "missing_key_defender": false }
  },
  "away": {
    "name": "客隊範例", "general_strength": { "home_advantage_weight": 0.9 },
    "offensive_stats": { "goals_scored_avg": 1.1, "xg_avg": 1.2 },
    "defensive_stats": { "goals_conceded_avg": 1.6, "xga_avg": 1.5 },
    "style_of_play": { "volatility": "high" },
    "context_modifiers": { "motivation": "normal", "missing_key_defender": true }
  }
}
"""

with tab1:
    json_text = st.text_area("在此貼上 JSON", value=default_json, height=250)
    if json_text:
        try:
            input_data = json.loads(json_text)
        except json.JSONDecodeError:
            st.error("JSON 格式錯誤，請檢查內容。")

with tab2:
    uploaded_file = st.file_uploader("選擇 .json 或 .txt 檔案", type=['json', 'txt'])
    if uploaded_file is not None:
        try:
            input_data = json.load(uploaded_file)
            st.success(f"成功讀取檔案：{uploaded_file.name}")
        except:
            st.error("檔案讀取失敗，請確認內容為有效 JSON。")

# --- 執行分析按鈕 ---
if st.button("🚀 開始分析", type="primary"):
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

        # --- Tab 分頁顯示詳細數據 ---
        res_tab1, res_tab2, res_tab3 = st.tabs(["📊 投注價值分析", "🧠 智能裁決", "🎯 波膽分佈"])

        candidates = []

        with res_tab1:
            # 1x2 表格
            st.subheader("💰 獨贏 (1x2) 分析")
            data_1x2 = []
            for tag, prob, key in [("主勝", prob_h, "home"), ("和局", prob_d, "draw"), ("客勝", prob_a, "away")]:
                odd = engine.market["1x2_odds"][key]
                ev = (prob * odd - 1) * 100 + market_bonus[key]
                # 計算獲利
                profit = (odd - 1) * unit_stake
                data_1x2.append([tag, f"{prob*100:.1f}%", odd, f"{ev:+.1f}%", f"${profit:.1f}"])
                if ev > 1.5:
                    candidates.append({"type":"1x2", "pick":tag, "ev":ev, "odds":odd, "prob":prob})
            
            df_1x2 = pd.DataFrame(data_1x2, columns=["選項", "模型機率", "賠率", "EV (期望值)", "預計獲利"])
            st.table(df_1x2)

            col_ah, col_ou = st.columns(2)
            
            # 亞盤
            with col_ah:
                st.subheader("🛡️ 亞盤 (Handicap)")
                data_ah = []
                for hcap in engine.market["handicaps"]:
                    ev = engine.ah_ev(M, hcap, engine.market["target_odds"]) + market_bonus["home"]
                    # 亞盤獲利 (假設賠率是 target_odds)
                    h_odd = engine.market["target_odds"]
                    profit = (h_odd - 1) * unit_stake
                    
                    data_ah.append([f"主 {hcap:+}", f"{ev:+.1f}%", f"${profit:.1f}"])
                    if ev > 2:
                        candidates.append({"type":"AH", "pick":f"主 {hcap:+}", "ev":ev, "odds":h_odd, "prob":0.5+ev/200})
                st.table(pd.DataFrame(data_ah, columns=["盤口", "EV", "預計獲利"]))

            # 大小球
            with col_ou:
                st.subheader("📐 大小球 (Over/Under)")
                data_ou = []
                for line in engine.market["goal_lines"]:
                    o_prob = sum(M[i,j] for i in range(9) for j in range(9) if i+j>line)
                    ev_o = (o_prob * engine.market["target_odds"] - 1) * 100
                    # 大小球獲利
                    o_odd = engine.market["target_odds"]
                    profit = (o_odd - 1) * unit_stake
                    
                    data_ou.append([f"大 {line}", f"{o_prob*100:.1f}%", f"{ev_o:+.1f}%", f"${profit:.1f}"])
                    if ev_o > 2:
                        candidates.append({"type":"OU", "pick":f"大 {line}", "ev":ev_o, "odds":o_odd, "prob":o_prob})
                st.table(pd.DataFrame(data_ou, columns=["盤口", "機率", "EV", "預計獲利"]))

            # 最終推薦列表
            st.subheader("📝 最佳投資組合 (Top Picks)")
            if candidates:
                final_list = sorted(candidates, key=lambda x:x["ev"], reverse=True)
                reco_data = []
                for p in final_list[:3]:
                    kelly = calc_kelly(p["prob"], p["odds"], kelly_frac)
                    # 計算該選項的預計獲利
                    profit = (p['odds'] - 1) * unit_stake
                    reco_data.append([f"[{p['type']}] {p['pick']}", p['odds'], f"{p['ev']:+.1f}%", f"{kelly:.1f}%", f"${profit:.1f}"])
                
                st.dataframe(pd.DataFrame(reco_data, columns=["選項", "賠率", "EV", "建議注碼%", "預計獲利 (單注)"]), use_container_width=True)
                st.caption(f"* 預計獲利是基於您設定的單注本金 ${unit_stake} 計算")
            else:
                st.info("目前無高 EV 選項推薦。")

        with res_tab2:
            st.subheader("🧠 模型裁決與警報")
            
            # 節奏裁決
            total_xg = lh + la
            if total_xg > 3.5:
                st.warning(f"🟠 高變異節奏 (Total xG: {total_xg:.2f}) - 攻防轉換快，紅牌點球影響大。")
            elif total_xg > 2.5:
                st.success(f"🟢 中性節奏 (Total xG: {total_xg:.2f}) - 模型穩定性佳。")
            else:
                st.info(f"🔵 低節奏 (Total xG: {total_xg:.2f}) - 爆冷多來自定位球。")

            # 智能市場檢核
            if candidates:
                top_pick = sorted(candidates, key=lambda x:x["ev"], reverse=True)[0]
                market_implied = 1.0 / top_pick['odds']
                model_prob = top_pick['prob']
                edge_diff = model_prob - market_implied
                
                st.markdown("---")
                st.write(f"**最佳選項 [{top_pick['pick']}] 深度檢核：**")
                col_c1, col_c2 = st.columns(2)
                col_c1.metric("模型機率", f"{model_prob*100:.1f}%")
                col_c2.metric("市場隱含機率", f"{market_implied*100:.1f}%")

                if edge_diff < 0:
                    st.error("🔴 虛高風險 (High Odds Trap)：EV 來自高賠率槓桿，實際勝率低。建議減半注碼。")
                elif edge_diff < 0.03:
                    st.warning("🟠 邊際優勢 (Thin Edge)：優勢不明顯，嚴格遵守注碼，不追單。")
                else:
                    st.success("🟢 真實價值 (True Value)：發現顯著機率偏差，信心買入。")
            
            # 相關性保護
            if len(candidates) >= 2:
                final_list = sorted(candidates, key=lambda x:x["ev"], reverse=True)
                p1 = final_list[0]
                p2 = final_list[1]
                
                def get_dir(name):
                    if "主" in name: return "HOME"
                    if "客" in name: return "AWAY"
                    if "大" in name: return "OVER"
                    return "NONE"
                
                if get_dir(p1['pick']) != "NONE" and get_dir(p1['pick']) == get_dir(p2['pick']):
                    st.error(f"⚠️ 資金控管警報：Top 1 [{p1['pick']}] 與 Top 2 [{p2['pick']}] 方向重疊！建議分攤注碼或只選其一。")

        with res_tab3:
            st.subheader("🎯 波膽 (Correct Score) 熱力圖")
            # 轉換矩陣為 DataFrame 以便顯示
            df_cs = pd.DataFrame(M[:6, :6], columns=[f"客 {j}" for j in range(6)], index=[f"主 {i}" for i in range(6)])
            # 格式化為百分比
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
                except:
                    pass
