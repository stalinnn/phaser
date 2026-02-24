import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import NMF
from scipy.special import softmax
import time
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="市场注意力拓扑监控台",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .big-number {
        font-size: 3em;
        font-weight: bold;
        color: #4CAF50;
    }
    .big-number-danger {
        color: #FF5252;
    }
    .big-number-warning {
        color: #FFC107;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC (ADIABATIC ATTENTION) ---
class AdiabaticAttentionV2:
    def __init__(self, n_components=5):
        self.n_components = n_components
        
    def fit_step(self, returns_window):
        if returns_window.shape[1] < 5: return None, None
        
        # 1. Correlation (Instantaneous)
        corr_mat = returns_window.corr().values
        corr_mat = np.nan_to_num(corr_mat)
        
        # 2. NMF (Extract Latent Factors K)
        # Shift to make positive
        A_pos = corr_mat + 1.0
        try:
            model = NMF(n_components=self.n_components, init='nndsvd', random_state=42, max_iter=200)
            W = model.fit_transform(A_pos)
            H = model.components_
            K = H.T
        except:
            K = np.random.rand(corr_mat.shape[0], self.n_components)

        # 3. Solve for Q (Attention)
        lambda_reg = 0.5
        try:
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            Q_linear = np.random.rand(corr_mat.shape[0], self.n_components)
            
        # 4. Temperature Scaling & Softmax
        # T ~ 1 / MaxCorrelation
        max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
        T_eff = 1.0 / (max_corr + 1e-6)
        T_eff = np.clip(T_eff, 0.1, 5.0)
        
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        
        return Q_nonlinear, T_eff

def calculate_parisi_order(Q):
    if Q is None: return 0
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

# --- DATA FETCHING ---
@st.cache_data(ttl=600) # Cache ticker list for 10 mins
def get_tickers():
    # S&P 100 Proxy (Top liquid stocks)
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
        "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG",
        "HD", "MA", "CVX", "ABBV", "ADBE", "NFLX", "DIS", "CMCSA", "TXN", "PM", 
        "HON", "QCOM", "AMGN", "CAT", "SPGI", "MS", "BA", "MMM", "T", "VZ"
    ]
    return list(set(tickers))

def fetch_live_data(tickers):
    # Fetch 1-minute data for the last 1 day (or max allowed)
    try:
        data = yf.download(tickers, period="1d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
        
        # Drop tickers with too many NaNs
        close = close.dropna(axis=1, thresh=int(len(close)*0.8))
        close = close.ffill().bfill()
        return close
    except Exception as e:
        st.error(f"数据获取错误: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    window_size = st.slider("滚动窗口 (K线数)", 10, 60, 30)
    n_components = st.slider("潜在因子数 (K)", 3, 10, 5)
    
    st.markdown("---")
    st.markdown("""
    **模型：** 逆向注意力机制 (Inverse Attention)
    **核心指标：** Parisi 拓扑序参量 ($q$)
    
    **指标解读：**
    - 🟢 **$q < 0.2$**: 安全 (注意力分散)
    - 🟠 **$q > 0.4$**: 警戒 (注意力凝聚)
    - 🔴 **$q > 0.6$**: 危险 (隧道视野 / 崩盘临界)
    """)
    
    if st.button("立即刷新数据"):
        st.cache_data.clear()

# --- MAIN LAYOUT ---
st.title("🦅 市场注意力拓扑实时监控台")
st.markdown("基于 **绝热注意力动力学 (Adiabatic Attention Dynamics)** 的系统性风险实时监测系统。")

# 1. Fetch & Process
with st.spinner("正在连接市场数据流..."):
    tickers = get_tickers()
    data = fetch_live_data(tickers)

if data is not None and len(data) > window_size:
    # Calculate Metrics for the LATEST window
    returns = np.log(data / data.shift(1)).dropna()
    latest_window = returns.iloc[-window_size:]
    
    solver = AdiabaticAttentionV2(n_components=n_components)
    Q, T_eff = solver.fit_step(latest_window)
    parisi_q = calculate_parisi_order(Q)
    
    # --- DASHBOARD ROW 1: KPI ---
    col1, col2, col3 = st.columns(3)
    
    # Determine Status Color
    status_color = "green"
    status_text = "正常 (NORMAL)"
    if parisi_q > 0.4:
        status_color = "orange" 
        status_text = "警戒 (WARNING)"
    if parisi_q > 0.6: 
        status_color = "red"
        status_text = "危险 (CRITICAL)"
        
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #888;">Parisi 拓扑序参量 ($q$)</div>
            <div class="big-number" style="color: {status_color};">{parisi_q:.4f}</div>
            <div style="font-weight: bold; color: {status_color};">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #888;">有效温度 ($T$)</div>
            <div class="big-number">{T_eff:.2f}</div>
            <div>流动性代理指标</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Calculate market trend (last bar vs window mean)
        market_last = data.iloc[-1].mean()
        market_prev = data.iloc[-2].mean()
        delta = (market_last - market_prev) / market_prev * 100
        color = "#4CAF50" if delta >= 0 else "#FF5252"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #888;">市场动量 (5分钟)</div>
            <div class="big-number" style="color: {color};">{delta:+.2f}%</div>
            <div>平均收益率</div>
        </div>
        """, unsafe_allow_html=True)

    # --- DASHBOARD ROW 2: CHARTS ---
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 拓扑序参量演化趋势 (过去4小时)")
        
        # Calculate Rolling History for Chart
        # (This might be slow, so we limit to last 50 points)
        history_len = 50
        history_q = []
        history_dates = []
        
        for i in range(len(returns) - history_len, len(returns)):
            w = returns.iloc[i-window_size:i]
            if len(w) < window_size: continue
            q_tmp, _ = solver.fit_step(w)
            p_tmp = calculate_parisi_order(q_tmp)
            history_q.append(p_tmp)
            history_dates.append(returns.index[i])
            
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=history_dates, y=history_q, mode='lines', name='Parisi Order', line=dict(color='#00CC96', width=3)))
        # Add Thresholds
        fig_ts.add_hline(y=0.4, line_dash="dot", line_color="orange", annotation_text="警戒线")
        fig_ts.add_hline(y=0.6, line_dash="dot", line_color="red", annotation_text="危险线")
        
        fig_ts.update_layout(
            title="拥挤度水平 ($q$) 随时间变化",
            yaxis_title="拥挤度 ($q$)",
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with col_right:
        st.subheader("🎯 注意力热力图")
        
        # Extract Attention Scores for current step
        # Sum Q across components for each stock
        if Q is not None:
            stock_attention = np.sum(Q, axis=1)
            # Create DF
            att_df = pd.DataFrame({
                '代码': latest_window.columns,
                '注意力权重': stock_attention
            }).sort_values('注意力权重', ascending=True).tail(15) # Top 15
            
            fig_bar = px.bar(
                att_df, 
                x='注意力权重', 
                y='代码', 
                orientation='h',
                title="Top 15 最拥挤资产",
                template="plotly_dark",
                color='注意力权重',
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            
    # --- ROW 3: NETWORK VIZ (Optional simplified) ---
    st.markdown("---")
    st.subheader("🕸️ 拓扑矩阵 (相关性 vs 注意力)")
    
    # Show the Q matrix heatmap
    if Q is not None:
        fig_map = px.imshow(
            Q.T, 
            labels=dict(x="资产", y="潜在因子", color="注意力权重"),
            x=latest_window.columns,
            y=[f"因子 {i+1}" for i in range(n_components)],
            title="投资者注意力分布 (Q矩阵)",
            template="plotly_dark",
            aspect="auto"
        )
        st.plotly_chart(fig_map, use_container_width=True)

else:
    st.warning("数据不足，无法计算拓扑指标。请等待市场开盘或增加数据周期。")

# Footer
st.markdown("---")
st.caption(f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据来源：Yahoo Finance (延迟15分钟)")
