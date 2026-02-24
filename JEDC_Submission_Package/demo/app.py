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
import os
import concurrent.futures
import akshare as ak  # Added for CN/US data without proxy
import requests # Added for Tencent API

# --- PATH CONFIG ---
# Get the absolute path of the current file (app.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOGGING UTILS ---
LOG_FILE = os.path.join(CURRENT_DIR, "debug_log.txt")

def log_to_file(message):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except: pass

def load_csv_cached(path, mtime=None):
    """
    Direct CSV loader. No Streamlit overhead.
    Fastest for files < 50MB.
    """
    try:
        log_to_file(f"Attempting to read CSV: {os.path.basename(path)}")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True, engine='c')
        except:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
        # Ensure index is datetime (critical)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # FORCE TIMEZONE NAIVE to avoid mismatches
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Deduplicate index (keep last)
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep='last')]
            
        df = df.sort_index() # Ensure sorted
        
        log_to_file(f"SUCCESS: Loaded {len(df)} rows from {os.path.basename(path)}")
        return df
    except Exception as e:
        log_to_file(f"ERROR: Read failed for {path}: {e}")
        return pd.DataFrame()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TFI 市场微观结构雷达",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .big-number {
        font-size: 2.5em;
        font-weight: bold;
        color: #4CAF50;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-sub {
        font-size: 0.8em;
        color: #666;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    /* Status Colors */
    .status-normal { color: #4CAF50; }
    .status-warning { color: #FFC107; }
    .status-danger { color: #FF5252; }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC (ADIABATIC ATTENTION) ---
class AdiabaticAttentionV2:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.last_T = 1.0
        
    def fit_step(self, returns_window, market_type="US_SP100"):
        # Allow fewer assets (e.g. 4 for fallback data)
        if returns_window.shape[1] < 3: return None, None
        
        # 1. Correlation (Instantaneous)
        try:
            corr_mat = returns_window.corr().values
            corr_mat = np.nan_to_num(corr_mat)
        except:
            return None, None
            
        # 2. NMF (Extract Latent Factors K)
        # Shift to make positive for NMF
        A_pos = corr_mat + 1.0
        
        # Robustness: Adjust K if we have fewer assets than requested components
        curr_k = min(self.n_components, A_pos.shape[0])
        
        try:
            # Increased max_iter to 1000 to avoid ConvergenceWarning
            model = NMF(n_components=curr_k, init='nndsvd', random_state=42, max_iter=1000)
            W = model.fit_transform(A_pos)
            H = model.components_
            K = H.T
        except:
            # Fallback if NMF fails
            K = np.random.rand(corr_mat.shape[0], curr_k)

        # 3. Solve for Q (Attention)
        lambda_reg = 0.5
        try:
            # Ridge regression to find Q such that Q*K^T approx Correlation
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(curr_k))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            Q_linear = np.random.rand(corr_mat.shape[0], curr_k)
            
        # 4. Temperature Scaling & Softmax
        # T ~ 1 / MaxCorrelation (inverse of strongest link)
        # We use off-diagonal max to avoid self-correlation=1
        np.fill_diagonal(corr_mat, 0)
        max_corr = np.max(np.abs(corr_mat))
        
        # Adaptive Scaling based on Market Type
        if market_type == "CRYPTO":
            # Crypto is naturally highly correlated, so we reduce sensitivity
            # Base Temp = 1 / (max_corr * 2.0 + 0.5) -> Higher T
            scale_factor = 2.0
            offset = 0.5
        else:
            # Stock markets (US/CN) need high sensitivity to detect rare events
            scale_factor = 5.0
            offset = 0.1
            
        T_eff = 1.0 / (max_corr * scale_factor + offset) 
        T_eff = np.clip(T_eff, 0.05, 5.0)
        self.last_T = T_eff
        
        # Attention Mechanism: Softmax(Q * K.T / T)
        # Here we simplify to Softmax(Q_linear / T) for visualization of Q distribution
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        
        return Q_nonlinear, T_eff

def calculate_parisi_order(Q):
    if Q is None: return 0
    # Q is (N_assets, N_factors)
    # Overlap matrix: O_ij = \sum_k Q_ik * Q_jk
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    # Parisi parameter q is the mean of off-diagonal overlaps
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_tickers(market_type):
    if market_type == "US_SP100":
        # S&P 100 Proxy (Top liquid stocks)
        return [
            "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
            "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
            "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG",
            "HD", "MA", "CVX", "ABBV", "ADBE", "NFLX", "DIS", "CMCSA", "TXN", "PM", 
            "HON", "QCOM", "AMGN", "CAT", "SPGI", "MS", "BA", "MMM", "T", "VZ"
        ]
    elif market_type == "CN_A50":
        # China A50 Proxies (using Yahoo Finance suffixes .SS and .SZ)
        return [
            "600519.SS", "601318.SS", "600036.SS", "600276.SS", "601012.SS",
            "600900.SS", "600887.SS", "600030.SS", "603288.SS", "601888.SS",
            "000858.SZ", "000333.SZ", "002415.SZ", "002594.SZ", "000651.SZ",
            "000002.SZ", "000725.SZ", "000001.SZ", "601166.SS", "601328.SS"
        ]
    elif market_type == "CRYPTO":
        return [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
            "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD"
        ]
    return []

import akshare as ak  # Added for CN/US data without proxy

# @st.cache_data(ttl=3600)  <-- Removed to fix CacheReplayClosureError with st.progress
def fetch_history_data(tickers, period="2y"):
    """
    Fetch data with Fallback mechanism.
    Hybrid Source: 
    - AkShare (Sina/EastMoney) for No-Proxy access in China (A-Share & US-Share).
    - YFinance as backup.
    """
    data = None
    used_fallback = False
    
    # Calculate dates for AkShare
    end_date = datetime.now()
    if period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "2y":
        start_date = end_date - timedelta(days=365*2)
    elif period == "5y":
        start_date = end_date - timedelta(days=365*5)
    else:
        start_date = datetime(2000, 1, 1) # Max/Default

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # Helper to fetch single stock from AkShare
    def fetch_single_ak(symbol, market_type, start_date_str, end_date_str):
        try:
            df = pd.DataFrame()
            if market_type == "CN":
                # A-Share: 东方财富接口
                clean_sym = symbol.split('.')[0] 
                df = ak.stock_zh_a_hist(symbol=clean_sym, period="daily", start_date=start_date_str, end_date=end_date_str, adjust="qfq")
                if not df.empty:
                    df = df.rename(columns={'日期': 'Date', '收盘': symbol})
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    return df[[symbol]] 
            elif market_type == "US":
                # US-Share: 腾讯财经接口 (纯HTTP, 支持多线程)
                # 腾讯接口无需 JS 引擎，速度极快
                import requests
                clean_sym = symbol.replace("-USD", "").lower()
                # 腾讯美股前缀通常是 us
                url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=us{clean_sym},day,,,320,qfq"
                r = requests.get(url, timeout=2)
                data = r.json()
                kline = data['data'][f'us{clean_sym}']['day']
                
                # Parse
                if not kline: return None # Handle empty response
                
                temp_df = pd.DataFrame(kline)
                if temp_df.empty or temp_df.shape[1] < 3: return None
                
                # Columns: date, open, close, high, low, volume
                temp_df = temp_df.iloc[:, [0, 2]] # Date and Close
                temp_df.columns = ['Date', symbol]
                temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                temp_df[symbol] = pd.to_numeric(temp_df[symbol])
                temp_df.set_index('Date', inplace=True)
                
                # Filter range
                mask = (temp_df.index >= pd.Timestamp(start_date_str)) & (temp_df.index <= pd.Timestamp(end_date_str))
                return temp_df[mask]

        except Exception as e:
            log_to_file(f"Fetch Error {symbol}: {e}")
            pass
        return None

    # --- INCREMENTAL UPDATE LOGIC ---
    # 1. Define Warehouse Path
    warehouse_dir = os.path.join(CURRENT_DIR, "data", "warehouse")
    if not os.path.exists(warehouse_dir): os.makedirs(warehouse_dir)
    
    # Identify cache file based on market and tickers hash (simple version: just market name)
    # Assuming tickers list doesn't change often. If it does, we need smarter merging.
    # For this demo, we use market name.
    if '.SS' in tickers[0] or '.SZ' in tickers[0]: cache_name = "CN_A50.csv"
    elif '-USD' in tickers[0]: cache_name = "CRYPTO.csv"
    else: cache_name = "US_SP100.csv"
    
    cache_path = os.path.join(warehouse_dir, cache_name)
    
    # 2. Load Local Cache
    local_data = None
    start_fetch_date = datetime(2000, 1, 1) # Default if no cache
    
    if os.path.exists(cache_path):
        try:
            log_to_file(f"Found local warehouse: {cache_path}")
            local_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not local_data.empty:
                last_date = local_data.index[-1]
                # Start fetching from next day
                start_fetch_date = last_date + timedelta(days=1)
                #st.toast(f"已加载本地历史数据 (截至 {last_date.date()})", icon="📂")
                log_to_file(f"Local warehouse loaded. Last date: {last_date.date()}")
        except:
            local_data = None
            log_to_file("Error loading local warehouse")
            
    # Check if we need to fetch anything
    today = datetime.now()
    if start_fetch_date.date() > today.date():
        # Cache is up to date (or from future?), no fetch needed
        log_to_file("Warehouse up-to-date. No fetch needed.")
        
        # Apply Final Cleaning before returning!
        close = local_data
        if close is not None:
            close = close.dropna(axis=1, thresh=int(len(close)*0.8))
            close = close.ffill().bfill()
            log_to_file(f"DEBUG CACHE-HIT POST-CLEAN: Shape={close.shape}")
            
        return close, False
        
    log_to_file(f"Need update from {start_fetch_date.date()} to {today.date()}")
        
        # 3. Fetch Incremental Data
        # ... existing logic ...
    # STRATEGY: Hybrid Backfill
    # 1. Try to load long-term history from fallback CSV (2000-2024)
    # 2. Use that as base, and only fetch increment from its last date
    
    if local_data is None:
        # Try loading offline fallback first to get deep history
        fallback_map = {
            "CN": os.path.join(CURRENT_DIR, "data", "A_share_fallback.csv"),
            "US": os.path.join(CURRENT_DIR, "data", "US_SP100_fallback.csv"),
            "CRYPTO": os.path.join(CURRENT_DIR, "data", "CRYPTO_fallback.csv")
        }
        
        market_key = "US"
        if len(tickers) > 0:
            if '.SS' in tickers[0]: market_key = "CN"
            elif '-USD' in tickers[0]: market_key = "CRYPTO"
            
        fb_path = fallback_map.get(market_key)
        if os.path.exists(fb_path):
            try:
                log_to_file(f"Loading Base History from: {fb_path}")
                base_df = pd.read_csv(fb_path, index_col=0, parse_dates=True)
                if not base_df.empty:
                    base_df.index = pd.to_datetime(base_df.index)
                    local_data = base_df
                    # Start fetching from the day AFTER the base history ends
                    start_fetch_date = local_data.index[-1] + timedelta(days=1)
                    log_to_file(f"Base History Loaded. Range: {local_data.index[0].date()} to {local_data.index[-1].date()}")
            except Exception as e:
                log_to_file(f"Failed to load base history: {e}")
                
        # If still None (no fallback found), then we have to do full fetch
        if local_data is None:
             if period == "1y": start_fetch_date = today - timedelta(days=365)
             elif period == "2y": start_fetch_date = today - timedelta(days=365*2)
             elif period == "5y": start_fetch_date = today - timedelta(days=365*5)
             log_to_file(f"No base history found. Full fetch from {start_fetch_date.date()}")
    
    start_str = start_fetch_date.strftime("%Y%m%d")
    end_str = today.strftime("%Y%m%d")
    
    # 3. Fetch Incremental Data
    # Only fetch if there is a gap > 0 days
    is_incremental = (start_fetch_date.date() <= today.date())
    new_data = None
    
    if is_incremental:
        # ... (Fetch Logic using start_str, end_str) ...
        try:
            # Check explicit offline mode
            if os.environ.get("ST_OFFLINE_MODE") == "true":
                raise ConnectionError("Offline mode enforced")

            df_list = []
            valid_tickers = []
            
            # Determine Market Type roughly
            sample = tickers[0]
            if '.SS' in sample or '.SZ' in sample: current_market = "CN"
            elif '-USD' in sample: current_market = "CRYPTO" 
            else: current_market = "US"

            if current_market in ["CN", "US"]:
                # Since we switched US to Tencent (HTTP), both CN and US are now thread-safe!
                use_parallel = True 
                
                if use_parallel:
                    # ThreadPool is safe now for both markets
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                        def fetch_wrapper(ticker):
                            return fetch_single_ak(ticker, current_market, start_str, end_str), ticker

                        future_to_ticker = {executor.submit(fetch_wrapper, t): t for t in tickers}
                        total = len(tickers)
                        completed = 0
                        
                        # Only show progress if fetching > 5 days or > 10 tickers
                        show_progress = (today - start_fetch_date).days > 5
                        if show_progress:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        for future in concurrent.futures.as_completed(future_to_ticker):
                            stock_df, t = future.result()
                            if stock_df is not None:
                                df_list.append(stock_df)
                                valid_tickers.append(t)
                            if show_progress:
                                completed += 1
                                progress_bar.progress(completed / total)
                                status_text.caption(f"正在增量更新... ({completed}/{total})")
                        
                        if show_progress:
                            progress_bar.empty()
                            status_text.empty()
                else:
                    # US Market: Sequential
                    total = len(tickers)
                    show_progress = (today - start_fetch_date).days > 5
                    if show_progress:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                    for i, t in enumerate(tickers):
                        if show_progress: status_text.caption(f"正在更新美股... ({i+1}/{total})")
                        log_to_file(f"Fetching US ticker: {t}")
                        
                        try:
                            # Add a simple timeout mechanism if possible? 
                            # AkShare doesn't support timeout arg natively in all funcs.
                            # We just log it.
                            start_t = time.time()
                            stock_df = fetch_single_ak(t, current_market, start_str, end_str)
                            elapsed = time.time() - start_t
                            if elapsed > 2.0: log_to_file(f"SLOW FETCH: {t} took {elapsed:.2f}s")
                            
                            if stock_df is not None:
                                df_list.append(stock_df)
                        except Exception as e:
                            log_to_file(f"FAILED FETCH: {t} - {e}")
                            
                        if show_progress: progress_bar.progress((i + 1) / total)
                    
                    if show_progress:
                        progress_bar.empty()
                        status_text.empty()
                
                if df_list:
                    new_data = pd.concat(df_list, axis=1)
                
            else:
                 # Crypto -> Fallback to YFinance (Full fetch usually)
                 pass # Not implemented for incremental yet

        except Exception as e:
            st.error(f"Update failed: {e}")
    
    # 4. Merge & Save
    final_data = None
    
    if new_data is not None and not new_data.empty:
        if local_data is not None:
            # Combine
            # Align columns first
            # 1. Reindex new_data to match local_data columns (add missing cols as NaN)
            # 2. Reindex local_data to match new_data columns? 
            # Best: Concat and combine_first
            
            # Simple concat (assuming tickers mostly same)
            combined = pd.concat([local_data, new_data])
            # Remove duplicates index
            final_data = combined[~combined.index.duplicated(keep='last')].sort_index()
        else:
            final_data = new_data
            
        # Save back to warehouse
        final_data.to_csv(cache_path)
        if is_incremental and local_data is not None:
            st.toast(f"更新完成！新增 {len(new_data)} 天数据", icon="✅")
            
    else:
        # No new data or fetch failed
        final_data = local_data
    
    # If we still have no data (fresh run + fetch failed), try old fallback
    if final_data is None or final_data.empty:
         # ... (Existing Fallback Logic) ...
         # Check and Load CSV Fallback if online methods failed
         fallback_map = {
             "CN": os.path.join(CURRENT_DIR, "data", "A_share_fallback.csv"),
             "US": os.path.join(CURRENT_DIR, "data", "US_SP100_fallback.csv"),
             "CRYPTO": os.path.join(CURRENT_DIR, "data", "CRYPTO_fallback.csv")
         }
         # ... (Load fallback)
         try:
             # Simplified fallback loading
             if '.SS' in tickers[0]: k="CN"
             elif '-USD' in tickers[0]: k="CRYPTO"
             else: k="US"
             fb_path = fallback_map.get(k)
             if os.path.exists(fb_path):
                 final_data = pd.read_csv(fb_path, index_col=0, parse_dates=True)
                 used_fallback = True
                 log_to_file(f"Loaded from Fallback CSV: {fb_path}")
         except: pass

    # --- FINAL EMERGENCY MOCK (To prevent App Crash) ---
    if (final_data is None or final_data.empty) and len(tickers) > 0:
        log_to_file("EMERGENCY: Generating Mock Data to prevent crash.")
        # Generate 2 years of mock data
        dates = pd.date_range(end=datetime.now(), periods=500, freq='B')
        mock_data = {}
        for t in tickers:
            # Random walk
            price = 100
            prices = [price]
            for _ in range(499):
                price *= (1 + np.random.normal(0, 0.02))
                prices.append(price)
            mock_data[t] = prices
        final_data = pd.DataFrame(mock_data, index=dates)
        used_fallback = True
        st.warning("⚠️ 无法获取真实数据，已启用【模拟数据模式】以演示功能。")

    close = final_data
    # Final Cleaning
    if close is not None:
        # Debug Log
        log_to_file(f"DEBUG PRE-CLEAN: Shape={close.shape}, IndexType={type(close.index)}")
        if not close.empty:
            log_to_file(f"DEBUG PRE-CLEAN Range: {close.index[0]} to {close.index[-1]}")
            
        close = close.dropna(axis=1, thresh=int(len(close)*0.8))
        close = close.ffill().bfill()
        
        log_to_file(f"DEBUG POST-CLEAN: Shape={close.shape}")
        
    return close, used_fallback

# --- SIDEBAR (Hidden/Collapsed by default for mobile friendliness) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/radar-plot.png", width=64)
    st.caption("v2.0.1 | Author: 徐明阳")

# --- MAIN LOGIC ---

col_header, col_manual = st.columns([6, 1])

with col_header:
    st.title(f"🦅 市场微观结构雷达") # Market name dynamic update moved below

with col_manual:
    # Use popover if available (Streamlit 1.33+) for better width
    try:
        manual_container = st.popover("📖 用户手册", use_container_width=True)
    except AttributeError:
        # Fallback for older versions
        manual_container = st.expander("📖 用户手册", expanded=False)
        
    with manual_container:
        st.markdown("""
        **1. 系统简介**
        本系统基于**统计物理与深度学习**的前沿算法，利用**逆向注意力机制 (Inverse Attention Mechanism)** 重构金融市场的微观资金流拓扑。它能穿透宏观价格噪音，直接观测市场参与者的群体行为同步性。
        
        **2. 操作指南**
        - **切换市场**: 选择 US (美股)/CN (A股)/CRYPTO (加密货币)。
        - **时间旅行**: 拖动滑块回溯历史，复盘历次危机。
        
        **3. 技术验证**
        """)
        
        st.markdown("---")
        st.markdown("**独特性检验 (Orthogonality)**")
        st.caption("与传统波动率指标低相关，提供增量信息")
        try:
            st.image(os.path.join(CURRENT_DIR, "figures", "Figure_1B_Heatmap.png"), use_container_width=True)
        except: pass
            
        st.markdown("---")
        st.markdown("**实战案例：2020 熔断**")
        st.caption("提前预警流动性枯竭导致的非线性崩盘")
        try:
            st.image(os.path.join(CURRENT_DIR, "figures", "event_study_2020.png"), use_container_width=True)
        except: pass
            
        st.markdown("---")
        st.markdown("**性能基准测试 (Benchmark)**")
        st.caption("相比传统指标，具有更高的召回率和提前量")
        try:
            st.image(os.path.join(CURRENT_DIR, "figures", "threshold_sensitivity.png"), use_container_width=True)
        except: pass
        
        st.markdown("""
        | 指标 (Metric) | TFI (Ours) | VIX/HV | AR |
        | :--- | :--- | :--- | :--- |
        | **召回率 (Recall)** | **91.8%** | 64.3% | 63.0% |
        | **准确率 (Precision)** | **6.9%** | 8.0% | 6.1% |
        | **提前量 (Days)** | **57.9** | 14.2 | 23.8 |
        | **特质** | **高灵敏** | 滞后 | 平滑 |
        """)
        st.caption("*注：数据基于 Threshold=0.25 的保守策略测算。AR=Absorption Ratio。")

# --- CONFIGURATION (Moved to Main Area for Mobile Friendliness) ---
with st.expander("⚙️ 监控配置 (Settings)", expanded=True):
    # Row 1: Basic Setup
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        mode = st.radio("模式选择", ["实时监控 (Live)", "历史回测 (Backtest)"], index=1, horizontal=True)
    with c2:
        market = st.selectbox("资产池", ["US_SP100", "CN_A50", "CRYPTO"], index=0)
    with c3:
        # Default changed to 45 based on Robustness Check (Fig 2)
        window_size = st.slider("滚动窗口 (Days)", 10, 90, 45, help="计算相关性矩阵的时间窗口长度")

    # Row 2: Advanced Parameters
    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        # Adaptive Max K based on actual data
        k_max = 10
        n_components = st.slider("潜在因子数 (K)", 3, k_max, 5, help="非负矩阵分解的秩，代表市场隐含的主题数")
    with c5:
        warn_thresh = st.number_input("警戒阈值 (Warning)", 0.2, 0.8, 0.4, 0.05)
    with c6:
        crit_thresh = st.number_input("熔断阈值 (Critical)", 0.2, 0.9, 0.65, 0.05)
        
    st.info("""
    **TFI 序参量 ($q$):** 量化市场微观状态的"铁磁有序度"。 $q \\to 0$: 安全; $q \\to 1$: 危险。
    """)
    
    st.divider()
    
    # Panorama settings (Only relevant if backtest/history used effectively)
    today = datetime.now().date()
    min_date = datetime(2000, 1, 1).date()
    default_start = min_date 
    
    panorama_range = st.date_input(
        "全景图时间范围",
        value=(default_start, today),
        min_value=min_date,
        max_value=today,
        help="选择顶部历史全景图的显示区间"
    )

if mode == "实时监控 (Live)":
    st.markdown("### 🔴 实时监控模式 (Live Market Data)")
    st.caption(f"数据源: Yahoo Finance | 最后更新: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.markdown("### 🕰️ 历史回溯模式 (Historical Replay)")
    st.caption("正在复盘历史市场状态，请使用下方滑块选择观测时间点。")

st.markdown(f"**资产池**: `{len(get_tickers(market))}` 支标的")

# Data Loading (Direct)
tickers = get_tickers(market)
with st.spinner("正在构建拓扑场..."):
    if mode == "实时监控 (Live)":
        raw_data, is_fallback = fetch_history_data(tickers, period="1y")
    else:
        raw_data, is_fallback = fetch_history_data(tickers, period="5y")

if raw_data is None or len(raw_data) < window_size:
    st.error("数据获取失败或数据量不足，请检查网络连接。")
    st.stop()

# Auto-adjust N_components if data is sparse (e.g. fallback mode with only 4 stocks)
n_assets = raw_data.shape[1]
if n_components >= n_assets:
    new_k = max(2, n_assets - 1)
    if new_k != n_components:
        st.warning(f"⚠️ 资产数量 ({n_assets}) 少于请求的因子数 ({n_components})。自动调整 K={new_k}。")
        n_components = new_k

if is_fallback:
    # Only show warning if NOT in explicit offline mode (to keep UI clean for cloud demo)
    if os.environ.get("ST_OFFLINE_MODE") != "true":
        st.warning(f"⚠️ 网络连接不稳定，当前显示【离线演示数据】(仿真生成)，仅供功能展示。")
    else:
        st.caption("ℹ️ 当前运行于离线演示模式 (Offline Demo Mode)")

# Calculate Returns
returns = np.log(raw_data / raw_data.shift(1)).dropna()
# Force TZ Naive
if returns.index.tz is not None:
    returns.index = returns.index.tz_localize(None)

# --- BACKTEST / SIMULATION MODE UI ---
if mode == "历史回测 (Backtest)":
    # Slider to select date
    dates = returns.index
    
    # Create options for the slider (must be hashable/list)
    # We skip the first 'window_size' days because we need enough data for the first window
    if len(dates) > window_size:
        available_dates = dates[window_size:].to_pydatetime()
        
        st.markdown("### 🕰️ 时间旅行控制台 (Time Travel)")
        st.info("拖动滑块以回溯历史市场状态（如：拖动至 2020-03 查看疫情熔断）")
        
        col_date1, col_date2 = st.columns([3, 1])
        
        with col_date1:
            # Use select_slider for coarse selection
            selected_date_slider = st.select_slider(
                "快速回溯",
                options=available_dates,
                value=available_dates[-1],
                format_func=lambda d: d.strftime('%Y-%m-%d')
            )
        
        with col_date2:
            # Date input for precise selection
            if 'selected_date_precise' not in st.session_state:
                 st.session_state.selected_date_precise = available_dates[-1]
                 
            selected_date_input = st.date_input(
                "精确日期",
                value=selected_date_slider,
                min_value=available_dates[0],
                max_value=available_dates[-1]
            )
        
        # Priority logic
        input_dt = datetime.combine(selected_date_input, datetime.min.time())
        nearest_date = min(available_dates, key=lambda x: abs(x - input_dt))
        
        if selected_date_slider != available_dates[-1]: 
             final_date = selected_date_slider
        else:
             final_date = nearest_date
             
        selected_date_idx = returns.index.get_loc(final_date)
        st.caption(f"当前观测日期: **{final_date.strftime('%Y-%m-%d')}**")
        
        analysis_window = returns.iloc[selected_date_idx-window_size+1 : selected_date_idx+1]
        current_price = raw_data.iloc[selected_date_idx]
        selected_date = final_date 
        
    else:
        st.error(f"数据不足以进行回测。需要至少 {window_size} 天的数据，当前仅有 {len(dates)} 天。")
        st.stop() 
    
else:
    # Live mode: just take the latest window
    analysis_window = returns.iloc[-window_size:]
    current_price = raw_data.iloc[-1]
    selected_date = returns.index[-1]
    dates = returns.index # Add this definition
    # For live mode, final_date is also the last date
    final_date = selected_date
    selected_date_idx = len(returns.index) - 1 

# --- COMPUTATION ---
solver = AdiabaticAttentionV2(n_components=n_components)
# Q, T_eff = solver.fit_step(analysis_window, market_type=market) # Moved to cached function below

@st.cache_data(show_spinner=False)
def get_current_tfi_cached(window_data, n_comp, m_type):
    # Wrapper to cache the single step calculation
    s = AdiabaticAttentionV2(n_components=n_comp)
    return s.fit_step(window_data, market_type=m_type)

Q, T_eff = get_current_tfi_cached(analysis_window, n_components, market)
parisi_q = calculate_parisi_order(Q)

@st.cache_data(show_spinner=False)
def calculate_long_term_tfi(returns_df, window_size, n_components, market_type):
    """
    Incremental TFI Calculation with Local Persistence.
    Avoids re-calculating the whole history every time.
    """
    # 1. Setup Cache Path
    warehouse_dir = os.path.join(CURRENT_DIR, "data", "warehouse")
    cache_file = os.path.join(warehouse_dir, f"TFI_{market_type}_W{window_size}_K{n_components}.csv")
    
    # 2. Load Existing TFI Cache
    if os.path.exists(cache_file):
        # Use cached loader based on mtime
        mtime = os.path.getmtime(cache_file)
        cached_df = load_csv_cached(cache_file, mtime)
    else:
        cached_df = pd.DataFrame()
        
    # 3. Identify Missing Dates
    # We want to compute TFI for every 'step' days (e.g. 5)
    step = 5
    required_indices = range(window_size, len(returns_df), step)
    required_dates = returns_df.index[required_indices]
    
    # Filter out dates we already have
    if not cached_df.empty:
        # Find dates in required_dates that are NOT in cached_df.index
        # Using set difference for speed
        existing_dates = set(cached_df.index)
        to_compute_indices = [i for i, d in zip(required_indices, required_dates) if d not in existing_dates]
    else:
        to_compute_indices = list(required_indices)
        
    # 4. Compute Missing Values
    if to_compute_indices:
        solver = AdiabaticAttentionV2(n_components)
        new_results = []
        new_dates = []
        
        # Determine if we should show a progress bar (only if lots to compute)
        show_prog = len(to_compute_indices) > 20
        if show_prog:
            prog = st.progress(0)
            status = st.empty()
            
        for i, idx in enumerate(to_compute_indices):
            w = returns_df.iloc[idx-window_size:idx]
            q_val, _ = solver.fit_step(w, market_type)
            tfi = calculate_parisi_order(q_val)
            
            new_results.append(tfi)
            new_dates.append(returns_df.index[idx])
            
            if show_prog and i % 5 == 0:
                prog.progress((i + 1) / len(to_compute_indices))
                status.caption(f"正在增量计算 TFI 全景... ({i+1}/{len(to_compute_indices)})")
                
        if show_prog:
            prog.empty()
            status.empty()
            
        # 5. Merge and Save
        new_df = pd.DataFrame({'TFI': new_results}, index=new_dates)
        if not cached_df.empty:
            final_df = pd.concat([cached_df, new_df])
            final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
        else:
            final_df = new_df
            
        final_df.to_csv(cache_file)
        cached_df = final_df
    
    # Return aligned lists
    # We need to return only the subset corresponding to current returns_df to avoid future leakage or mismatch
    # RELAXED MATCHING: Instead of strict isin(step=5), we just return everything in the date range.
    # This prevents "disappearing chart" issues if the step alignment shifts slightly.
    
    start_dt = returns_df.index[0]
    end_dt = returns_df.index[-1]
    
    # Ensure timezone naive
    if cached_df.index.tz is not None: cached_df.index = cached_df.index.tz_localize(None)
    
    mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
    result_df = cached_df[mask]
    
    log_to_file(f"Macro TFI: Returning {len(result_df)} points (Cache total: {len(cached_df)})")
    
    # Return DataFrame directly for better session state serialization
    return result_df # Returns df with index and 'TFI' column

# --- DASHBOARD HEADER ---
col1, col2, col3, col4 = st.columns(4)

# Determine Status
if parisi_q > crit_thresh:
    status = "⛔ 极度危险 (CRITICAL)"
    s_color = "#FF5252" # Red
elif parisi_q > warn_thresh:
    status = "⚠️ 风险积聚 (WARNING)"
    s_color = "#FFC107" # Amber
else:
    status = "✅ 市场健康 (SAFE)"
    s_color = "#4CAF50" # Green

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 5px solid {s_color}">
        <div class="metric-label">系统状态</div>
        <div class="big-number" style="color: {s_color}; font-size: 1.8em;">{status.split(' ')[1]}</div>
        <div class="metric-sub">{selected_date.strftime('%Y-%m-%d')}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TFI (Parisi $q$)</div>
        <div class="big-number" style="color: {s_color}">{parisi_q:.3f}</div>
        <div class="metric-sub">微观拥挤度 (0~1)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Temperature color logic: Low temp = Bad
    if T_eff is not None:
        t_color = "#FF5252" if T_eff < 0.5 else "#4CAF50"
        t_display = f"{T_eff:.2f}"
    else:
        t_color = "#888"
        t_display = "N/A"
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">有效温度 ($T_{{eff}}$)</div>
        <div class="big-number" style="color: {t_color}">{t_display}</div>
        <div class="metric-sub">市场深度代理指标</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Simple Volatility of the window
    vol = analysis_window.mean(axis=1).std() * np.sqrt(252)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">当前波动率 (Vol)</div>
        <div class="big-number" style="color: #FFF">{vol*100:.1f}%</div>
        <div class="metric-sub">年化标准差</div>
    </div>
    """, unsafe_allow_html=True)

# --- CHARTS ROW ---
# Add Macro View before Detailed Charts
if mode == "历史回测 (Backtest)":
    st.markdown("### 🔭 历史宏观全景")
    
    with st.spinner("正在计算历史全景数据..."):
        macro_df = calculate_long_term_tfi(returns, window_size, n_components, market)
        full_dates = macro_df.index
        full_tfi = macro_df['TFI'].values
        
    # Filter by user selected range
        if isinstance(panorama_range, tuple) and len(panorama_range) == 2:
            start_date, end_date = panorama_range
            # Convert to pd.Timestamp for comparison
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            # Filter logic
            mask = [(d >= start_ts and d <= end_ts) for d in full_dates]
            display_dates = [d for d, m in zip(full_dates, mask) if m]
            display_tfi = [t for t, m in zip(full_tfi, mask) if m]
            
            # log_to_file(f"Macro Display: {len(display_dates)} points (Range: {start_ts.date()} - {end_ts.date()})")
        else:
            # Fallback if range not fully selected
            display_dates = full_dates
            display_tfi = full_tfi
            
        if not display_dates:
             st.warning("⚠️ 该时间段内无 TFI 数据，请调整时间范围。")

        fig_macro = go.Figure()
        fig_macro.add_trace(go.Scatter(
            x=display_dates, y=display_tfi, 
            mode='lines', 
            name='长期拥挤度 (Weekly)',
            line=dict(color='#00CC96', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.1)'
        ))
        
        # Mark current selected date
        # Convert datetime to timestamp (ms) to avoid Plotly type error in summation
        vline_ts = final_date.timestamp() * 1000
        fig_macro.add_vline(x=vline_ts, line_dash="dash", line_color="white", annotation_text="当前观测点")
        
        fig_macro.add_hline(y=crit_thresh, line_dash="dot", line_color="#FF5252", annotation_text="危机线")
        
        fig_macro.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="",
            yaxis_title="TFI Index",
            hovermode="x unified"
        )
        st.plotly_chart(fig_macro, use_container_width=True)


st.markdown("### 📉 深度透视 (近期细节)")
tab1, tab2, tab3 = st.tabs(["时间序列分析", "微观结构拓扑", "风险归因"])

@st.cache_data(show_spinner=False)
def calculate_high_res_tfi(returns_df, window_size, n_components, market_type, start_idx, end_idx):
    """
    High-Resolution (Daily) TFI Calculation with Persistence.
    Used for the detailed zoom-in view.
    """
    # 1. Setup Cache Path
    warehouse_dir = os.path.join(CURRENT_DIR, "data", "warehouse")
    # High-Res cache
    cache_file = os.path.join(warehouse_dir, f"TFI_Daily_{market_type}_W{window_size}_K{n_components}.csv")
    
    # 2. Load Existing Cache
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        cached_df = load_csv_cached(cache_file, mtime)
    else:
        cached_df = pd.DataFrame()
        
    # 3. Identify Missing Dates in the requested range
    # We need daily points from start_idx to end_idx
    target_indices = range(start_idx, end_idx + 1)
    target_dates = returns_df.index[target_indices]
    
    if not cached_df.empty:
        existing_dates = set(cached_df.index)
        to_compute_indices = [i for i, d in zip(target_indices, target_dates) if d not in existing_dates]
    else:
        to_compute_indices = list(target_indices)
        
    # 4. Compute Missing
    if to_compute_indices:
        solver = AdiabaticAttentionV2(n_components)
        new_results = []
        new_dates = []
        
        # Only show progress if > 10 points (daily calc is fast but 120 points take time)
        show_prog = len(to_compute_indices) > 10
        if show_prog:
            prog = st.progress(0)
            status = st.empty()
            
        for i, idx in enumerate(to_compute_indices):
            # Boundary check
            if idx < window_size: continue
            
            w = returns_df.iloc[idx-window_size+1 : idx+1]
            q_val, _ = solver.fit_step(w, market_type)
            tfi = calculate_parisi_order(q_val)
            
            new_results.append(tfi)
            new_dates.append(returns_df.index[idx])
            
            if show_prog and i % 2 == 0:
                prog.progress((i + 1) / len(to_compute_indices))
                status.caption(f"正在补全高频 TFI... ({i+1}/{len(to_compute_indices)})")
                
        if show_prog:
            prog.empty()
            status.empty()
            
        # 5. Merge and Save
        new_df = pd.DataFrame({'TFI': new_results}, index=new_dates)
        if not cached_df.empty:
            final_df = pd.concat([cached_df, new_df])
            final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
        else:
            final_df = new_df
            
        final_df.to_csv(cache_file)
        cached_df = final_df

    # Return slice
    # Intersect
    mask = cached_df.index.isin(target_dates)
    result_df = cached_df[mask]
    return result_df.index, result_df['TFI'].values


with tab1:
    # Generate historical TFI for the plot (last 200 points up to selected date)
    # To save time, we might pre-compute or compute on the fly for a subset
    st.caption("加载最近 120 个交易日的 TFI 趋势...")

    lookback = 120
    start_plot = max(window_size, selected_date_idx - lookback)
    end_plot = selected_date_idx
    
    # Use the persistent function directly (Streamlit cache handles the rest)
    hist_dates, hist_q = calculate_high_res_tfi(returns, window_size, n_components, market, start_plot, end_plot)
    
    # Get corresponding prices (aligned by date)
    # raw_data might be full history, we need to match dates
    # Ensure raw_data index is tz-naive
    if raw_data.index.tz is not None: raw_data.index = raw_data.index.tz_localize(None)
    
    hist_prices = raw_data.loc[hist_dates].mean(axis=1)

    if len(hist_dates) > 0:
        # Dual Axis Plot
        fig = go.Figure()
    
        # Price Trace
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices, name="市场指数 (均值)",
            line=dict(color="#CCCCCC", width=1.5), # Changed from rgba(255,255,255,0.3) to solid light gray
            yaxis="y2"
        ))
    
        # TFI Trace
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_q, name="TFI (拥挤度)",
            line=dict(color=s_color, width=3),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(int(s_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}"
        ))
    
        # Thresholds
        fig.add_hline(y=warn_thresh, line_dash="dot", line_color="#FFC107", annotation_text="警戒线")
        fig.add_hline(y=crit_thresh, line_dash="dash", line_color="#FF5252", annotation_text="熔断线")
    
        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(title="TFI Index", range=[0, 1]),
            yaxis2=dict(title="Price Level", overlaying="y", side="right"),
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.subheader("注意力分配矩阵 ($Q$)")
        if Q is not None:
            # Heatmap of Assets vs Factors
            fig_map = px.imshow(
                Q.T,
                labels=dict(x="资产 (Assets)", y="隐因子 (Latent Factors)", color="权重"),
                x=analysis_window.columns,
                y=[f"F{i+1}" for i in range(Q.shape[1])],
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            fig_map.update_layout(height=350)
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("X轴为资产，Y轴为NMF提取的潜在风险因子。颜色越亮代表该资产被该因子'锁定'。")
        
    with col_t2:
        st.subheader("网络连通图 (Top 50 边)")
    
        # Check if Q is valid for visualization
        if Q is not None and len(Q.shape) == 2 and Q.shape[0] > 0:
            # Calculate full correlation for visualization
            corr_viz = analysis_window.corr().values
            # Create a graph
            edge_x = []
            edge_y = []
            node_x = []
            node_y = []
        
            # Simple circular layout
            n_nodes = len(analysis_window.columns)
            if n_nodes > 0:
                radius = 1
                angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                coords = {i: (radius*np.cos(a), radius*np.sin(a)) for i, a in enumerate(angles)}
            
                # Nodes
                for i in range(n_nodes):
                    node_x.append(coords[i][0])
                    node_y.append(coords[i][1])
                
                # Edges (only strong ones)
                # Threshold for drawing edge
                edge_thresh = 0.6
                np.fill_diagonal(corr_viz, 0)
                try:
                    strong_edges = np.argwhere(np.abs(corr_viz) > edge_thresh)
                    for i, j in strong_edges:
                        if i < j: # undirected
                            x0, y0 = coords[i]
                            x1, y1 = coords[j]
                            edge_x.append(x0)
                            edge_x.append(x1)
                            edge_x.append(None)
                            edge_y.append(y0)
                            edge_y.append(y1)
                            edge_y.append(None)
                except:
                    pass
                
                fig_net = go.Figure()
                fig_net.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
                fig_net.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hovertext=analysis_window.columns,
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=10,
                        color=np.sum(Q, axis=1), # Color by attention centrality
                        colorbar=dict(thickness=15, title='Centrality')
                    )
                ))
                fig_net.update_layout(
                    template="plotly_dark",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                    st.caption("暂无有效节点数据")
        else:
            st.caption("数据不足，无法生成网络图")

with tab3:
    st.subheader("谁在制造风险？(Risk Contributors)")
    if Q is not None:
        # Contribution = Sum of attention weights
        contrib = np.sum(Q, axis=1)
        contrib_df = pd.DataFrame({
            'Ticker': analysis_window.columns,
            'Attention_Score': contrib,
            'Last_Return': analysis_window.iloc[-1].values
        }).sort_values('Attention_Score', ascending=False)
    
        # Display top 10
        top_risk = contrib_df.head(10)
    
        fig_bar = px.bar(
            top_risk,
            x='Attention_Score',
            y='Ticker',
            orientation='h',
            color='Last_Return',
            color_continuous_scale='RdYlGn',
            title="Top 10 拥挤资产排行榜"
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)
    
        st.dataframe(contrib_df.style.background_gradient(subset=['Attention_Score'], cmap='Reds'), use_container_width=True)

# Footer
st.divider()
st.caption("© 2026 TFI Research Lab | Author: 徐明阳 | Powered by Adiabatic Attention Mechanism | Data: Yahoo Finance")
