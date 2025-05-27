
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os

# Error-free config loading with fallback
def load_config():
    """Load configuration with robust error handling and fallbacks"""
    try:
        # Try to import config.py
        from config import DEFAULT_STOCKS, DEFAULT_PARAMS, CHART_CONFIG, EXPORT_CONFIG
        return {
            'stocks': DEFAULT_STOCKS,
            'params': DEFAULT_PARAMS,
            'chart': CHART_CONFIG,
            'export': EXPORT_CONFIG
        }
    except ImportError:
        st.warning("⚠️ config.py not found. Using built-in default configuration.")
        # Fallback configuration if config.py doesn't exist
        return {
            'stocks': [
                'BBCA.JK', 'BMRI.JK', 'BBRI.JK', 'BBNI.JK', 'BRIS.JK',
                'TLKM.JK', 'ISAT.JK', 'EXCL.JK',
                'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'KLBF.JK', 'KAEF.JK',
                'ASII.JK', 'AUTO.JK', 'IMAS.JK',
                'ANTM.JK', 'INCO.JK', 'TINS.JK', 'PTBA.JK',
                'PGAS.JK', 'AKRA.JK',
                'BSDE.JK', 'LPKR.JK', 'PWON.JK',
                'GOTO.JK', 'BUKA.JK',
                'SMGR.JK', 'INTP.JK',
                'GGRM.JK', 'HMSP.JK'
            ],
            'params': {
                'weekly_rsi_min': 52,
                'weekly_adx_min': 18,
                'weekly_uo_min': 45,
                'daily_rsi_min': 50,
                'daily_rsi_max': 75,
                'volume_multiplier': 1.2,
                'market_cap_threshold': 500000000,
                'weekly_score_threshold': 7,
                'daily_score_threshold': 6
            },
            'chart': {
                'period': '3mo',
                'interval': '1d',
                'height': 600,
                'colors': {
                    'sma5': 'orange',
                    'sma20': 'red',
                    'rsi': 'purple',
                    'volume': 'blue'
                }
            },
            'export': {
                'filename_format': 'swing_screening_results_%Y%m%d_%H%M.csv',
                'columns_to_export': [
                    'Ticker', 'Price', 'Market_Cap', 'Volume_Ratio',
                    'Daily_RSI', 'Weekly_RSI', 'Weekly_ADX', 'Weekly_UO',
                    'Weekly_Score', 'Daily_Score', 'Total_Score'
                ]
            }
        }
    except Exception as e:
        st.error(f"❌ Error loading config.py: {str(e)}. Using fallback configuration.")
        # Return minimal fallback config
        return {
            'stocks': ['BBCA.JK', 'BMRI.JK', 'BBRI.JK', 'TLKM.JK', 'ASII.JK'],
            'params': {
                'weekly_rsi_min': 52, 'weekly_adx_min': 18, 'weekly_uo_min': 45,
                'daily_rsi_min': 50, 'daily_rsi_max': 75, 'volume_multiplier': 1.2,
                'market_cap_threshold': 500000000, 'weekly_score_threshold': 7,
                'daily_score_threshold': 6
            },
            'chart': {'period': '3mo', 'interval': '1d', 'height': 600, 'colors': {}},
            'export': {'filename_format': 'results_%Y%m%d_%H%M.csv', 'columns_to_export': []}
        }

# Load configuration
CONFIG = load_config()

# Konfigurasi halaman
st.set_page_config(
    page_title="Enhanced SWING Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .config-info {
        background-color: #e7f3ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #007bff;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 Enhanced SWING Screener</h1>', unsafe_allow_html=True)
st.markdown("**Algoritma Enhanced SWING Trading dengan Target Success Rate 81%+**")

# Config status indicator
config_source = "config.py" if 'config' in sys.modules else "Built-in defaults"
st.markdown(f'<div class="config-info">🔧 Configuration loaded from: <strong>{config_source}</strong></div>', 
            unsafe_allow_html=True)

# Sidebar untuk konfigurasi
st.sidebar.header("⚙️ Konfigurasi Screener")

# Input saham dengan fallback ke config
st.sidebar.subheader("📋 Daftar Saham")

# Convert config stocks to string format
default_stocks_text = "\n".join(CONFIG['stocks'])

stock_input = st.sidebar.text_area(
    "Masukkan kode saham (satu per baris):",
    value=default_stocks_text,
    height=200,
    help="Format: KODE.JK untuk saham Indonesia. Loaded from config.py"
)

# Parameter screening dengan nilai dari config
st.sidebar.subheader("🎯 Parameter Screening")

# Weekly parameters
st.sidebar.markdown("**Weekly Parameters:**")
weekly_rsi_min = st.sidebar.slider(
    "Weekly RSI Min", 
    45, 60, 
    CONFIG['params'].get('weekly_rsi_min', 52),
    help="Minimum RSI weekly untuk screening"
)
weekly_adx_min = st.sidebar.slider(
    "Weekly ADX Min", 
    15, 25, 
    CONFIG['params'].get('weekly_adx_min', 18),
    help="Minimum ADX weekly untuk trend strength"
)
weekly_uo_min = st.sidebar.slider(
    "Weekly UO Min", 
    40, 50, 
    CONFIG['params'].get('weekly_uo_min', 45),
    help="Minimum Ultimate Oscillator weekly"
)

# Daily parameters
st.sidebar.markdown("**Daily Parameters:**")
daily_rsi_min = st.sidebar.slider(
    "Daily RSI Min", 
    45, 55, 
    CONFIG['params'].get('daily_rsi_min', 50),
    help="Minimum RSI daily"
)
daily_rsi_max = st.sidebar.slider(
    "Daily RSI Max", 
    70, 80, 
    CONFIG['params'].get('daily_rsi_max', 75),
    help="Maximum RSI daily (avoid overbought)"
)
volume_multiplier = st.sidebar.slider(
    "Volume Multiplier", 
    1.0, 2.0, 
    CONFIG['params'].get('volume_multiplier', 1.2),
    help="Volume surge multiplier"
)

# Market cap filter
market_cap_options = ["100M", "500M", "1B", "5B", "10B"]
market_cap_values = [100e6, 500e6, 1e9, 5e9, 10e9]

# Find closest match to config value
config_market_cap = CONFIG['params'].get('market_cap_threshold', 500e6)
closest_index = min(range(len(market_cap_values)), 
                   key=lambda i: abs(market_cap_values[i] - config_market_cap))

market_cap_selection = st.sidebar.selectbox(
    "Market Cap Minimum",
    market_cap_options,
    index=closest_index,
    help="Minimum market capitalization filter"
)

market_cap_threshold = market_cap_values[market_cap_options.index(market_cap_selection)]

# Advanced parameters (collapsible)
with st.sidebar.expander("🔧 Advanced Parameters"):
    weekly_score_threshold = st.slider(
        "Weekly Score Threshold", 
        5, 9, 
        CONFIG['params'].get('weekly_score_threshold', 7),
        help="Minimum weekly criteria score"
    )
    daily_score_threshold = st.slider(
        "Daily Score Threshold", 
        4, 9, 
        CONFIG['params'].get('daily_score_threshold', 6),
        help="Minimum daily criteria score"
    )

# Fungsi untuk mengambil data saham dengan error handling
@st.cache_data(ttl=300)
def get_stock_data(ticker, period="6mo"):
    """Get stock data with robust error handling"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            st.warning(f"⚠️ No data available for {ticker}")
            return None, None

        info = stock.info
        return data, info
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {ticker}: {str(e)}")
        return None, None

# Fungsi untuk menghitung indikator weekly dengan error handling
def calculate_weekly_indicators(data):
    """Calculate weekly indicators with error handling"""
    try:
        # Konversi ke weekly
        weekly = data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(weekly) < 50:  # Need enough data
            return None

        # Indikator teknikal dengan error handling
        weekly['SMA5'] = ta.sma(weekly['Close'], length=5)
        weekly['SMA10'] = ta.sma(weekly['Close'], length=10)
        weekly['SMA20'] = ta.sma(weekly['Close'], length=20)
        weekly['SMA50'] = ta.sma(weekly['Close'], length=50)
        weekly['RSI'] = ta.rsi(weekly['Close'], length=14)

        # ADX with error handling
        try:
            adx_data = ta.adx(weekly['High'], weekly['Low'], weekly['Close'], length=14)
            weekly['ADX'] = adx_data['ADX_14'] if 'ADX_14' in adx_data.columns else np.nan
        except:
            weekly['ADX'] = np.nan

        # MACD with error handling
        try:
            macd_data = ta.macd(weekly['Close'])
            weekly['MACD'] = macd_data['MACD_12_26_9'] if 'MACD_12_26_9' in macd_data.columns else np.nan
            weekly['MACD_Signal'] = macd_data['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd_data.columns else np.nan
        except:
            weekly['MACD'] = np.nan
            weekly['MACD_Signal'] = np.nan

        # Ultimate Oscillator with error handling
        try:
            weekly['UO'] = ta.uo(weekly['High'], weekly['Low'], weekly['Close'])
        except:
            weekly['UO'] = np.nan

        return weekly
    except Exception as e:
        st.error(f"Error calculating weekly indicators: {str(e)}")
        return None

# Fungsi untuk menghitung indikator daily dengan error handling
def calculate_daily_indicators(data):
    """Calculate daily indicators with error handling"""
    try:
        daily = data.copy()

        if len(daily) < 50:  # Need enough data
            return None

        # Indikator teknikal dengan error handling
        daily['SMA5'] = ta.sma(daily['Close'], length=5)
        daily['SMA20'] = ta.sma(daily['Close'], length=20)
        daily['RSI'] = ta.rsi(daily['Close'], length=14)
        daily['Volume_SMA'] = ta.sma(daily['Volume'], length=20)

        # MACD with error handling
        try:
            macd_data = ta.macd(daily['Close'])
            daily['MACD'] = macd_data['MACD_12_26_9'] if 'MACD_12_26_9' in macd_data.columns else np.nan
            daily['MACD_Signal'] = macd_data['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd_data.columns else np.nan
        except:
            daily['MACD'] = np.nan
            daily['MACD_Signal'] = np.nan

        # ATR untuk volatility dengan error handling
        try:
            daily['ATR'] = ta.atr(daily['High'], daily['Low'], daily['Close'], length=14)
            daily['ATR_SMA'] = ta.sma(daily['ATR'], length=20)
        except:
            daily['ATR'] = np.nan
            daily['ATR_SMA'] = np.nan

        return daily
    except Exception as e:
        st.error(f"Error calculating daily indicators: {str(e)}")
        return None

# Fungsi screening weekly dengan error handling
def weekly_screening(weekly_data):
    """Weekly screening with robust error handling"""
    if weekly_data is None or len(weekly_data) < 2:
        return False, {}

    try:
        current = weekly_data.iloc[-1]
        previous = weekly_data.iloc[-2]

        criteria = {}

        # Safe comparison with NaN handling
        def safe_compare(val1, val2, operator='>'):
            try:
                if pd.isna(val1) or pd.isna(val2):
                    return False
                if operator == '>':
                    return val1 > val2
                elif operator == '<':
                    return val1 < val2
                else:
                    return False
            except:
                return False

        # 1. Higher highs and higher lows
        criteria['higher_high'] = safe_compare(current['High'], previous['High'])
        criteria['higher_low'] = safe_compare(current['Low'], previous['Low'])

        # 2. Price above SMA20
        criteria['price_above_sma20'] = safe_compare(current['Close'], current['SMA20'])

        # 3. SMA alignment
        criteria['sma_alignment'] = (
            safe_compare(current['SMA5'], current['SMA10']) and 
            safe_compare(current['SMA10'], current['SMA20'])
        )

        # 4. RSI bullish
        criteria['rsi_bullish'] = safe_compare(current['RSI'], weekly_rsi_min)

        # 5. ADX strength
        criteria['adx_strong'] = safe_compare(current['ADX'], weekly_adx_min)

        # 6. Ultimate Oscillator
        criteria['uo_bullish'] = safe_compare(current['UO'], weekly_uo_min)

        # 7. MACD bullish
        criteria['macd_bullish'] = safe_compare(current['MACD'], current['MACD_Signal'])

        # 8. Price above SMA50
        criteria['price_above_sma50'] = safe_compare(current['Close'], current['SMA50'])

        # 9. Weekly bullish candle
        criteria['weekly_bullish'] = safe_compare(current['Close'], current['Open'])

        # Hitung score
        score = sum(criteria.values())
        passed = score >= weekly_score_threshold

        return passed, criteria
    except Exception as e:
        st.error(f"Error in weekly screening: {str(e)}")
        return False, {}

# Fungsi screening daily dengan error handling
def daily_screening(daily_data, weekly_data):
    """Daily screening with robust error handling"""
    if daily_data is None or weekly_data is None or len(daily_data) < 20:
        return False, {}

    try:
        current = daily_data.iloc[-1]
        weekly_current = weekly_data.iloc[-1]

        criteria = {}

        # Safe comparison function
        def safe_compare(val1, val2, operator='>'):
            try:
                if pd.isna(val1) or pd.isna(val2):
                    return False
                if operator == '>':
                    return val1 > val2
                elif operator == '<':
                    return val1 < val2
                elif operator == 'between':
                    return val2[0] < val1 < val2[1]
                else:
                    return False
            except:
                return False

        # 1. Price above daily SMA20
        criteria['price_above_daily_sma20'] = safe_compare(current['Close'], current['SMA20'])

        # 2. Price above daily SMA5
        criteria['price_above_daily_sma5'] = safe_compare(current['Close'], current['SMA5'])

        # 3. Price above weekly SMA5
        criteria['price_above_weekly_sma5'] = safe_compare(current['Close'], weekly_current['SMA5'])

        # 4. Daily RSI range
        criteria['daily_rsi_range'] = safe_compare(current['RSI'], (daily_rsi_min, daily_rsi_max), 'between')

        # 5. Daily MACD bullish
        criteria['daily_macd_bullish'] = safe_compare(current['MACD'], current['MACD_Signal'])

        # 6. Volume surge
        criteria['volume_surge'] = safe_compare(current['Volume'], current['Volume_SMA'] * volume_multiplier)

        # 7. Bullish candle
        criteria['bullish_candle'] = safe_compare(current['Close'], current['Open'])

        # 8. Normal volatility
        criteria['normal_volatility'] = safe_compare(current['ATR_SMA'] * 1.8, current['ATR'])

        # 9. Strong candle body
        try:
            candle_range = current['High'] - current['Low']
            candle_body = abs(current['Close'] - current['Open'])
            criteria['strong_body'] = (candle_body / candle_range) > 0.4 if candle_range > 0 else False
        except:
            criteria['strong_body'] = False

        # Hitung score
        score = sum(criteria.values())
        passed = score >= daily_score_threshold

        return passed, criteria
    except Exception as e:
        st.error(f"Error in daily screening: {str(e)}")
        return False, {}

# Fungsi utama screening dengan progress tracking
def run_screening(stock_list):
    """Main screening function with progress tracking"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_stocks = len(stock_list)
    processed = 0
    errors = 0

    for i, ticker in enumerate(stock_list):
        status_text.text(f"Memproses {ticker}... ({i+1}/{total_stocks})")

        try:
            # Ambil data
            data, info = get_stock_data(ticker)

            if data is None or len(data) < 100:
                errors += 1
                continue

            # Filter market cap
            market_cap = info.get('marketCap', 0) if info else 0
            if market_cap < market_cap_threshold:
                continue

            # Hitung indikator
            weekly_data = calculate_weekly_indicators(data)
            daily_data = calculate_daily_indicators(data)

            if weekly_data is None or daily_data is None:
                errors += 1
                continue

            # Screening weekly
            weekly_passed, weekly_criteria = weekly_screening(weekly_data)

            if weekly_passed:
                # Screening daily
                daily_passed, daily_criteria = daily_screening(daily_data, weekly_data)

                if daily_passed:
                    current_price = daily_data['Close'].iloc[-1]
                    current_volume = daily_data['Volume'].iloc[-1]
                    avg_volume = daily_data['Volume_SMA'].iloc[-1]

                    # Safe division for volume ratio
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                    results.append({
                        'Ticker': ticker,
                        'Price': current_price,
                        'Market_Cap': market_cap,
                        'Volume_Ratio': volume_ratio,
                        'Daily_RSI': daily_data['RSI'].iloc[-1],
                        'Weekly_RSI': weekly_data['RSI'].iloc[-1],
                        'Weekly_ADX': weekly_data['ADX'].iloc[-1],
                        'Weekly_UO': weekly_data['UO'].iloc[-1],
                        'Weekly_Score': sum(weekly_criteria.values()),
                        'Daily_Score': sum(daily_criteria.values()),
                        'Total_Score': sum(weekly_criteria.values()) + sum(daily_criteria.values()),
                        'Weekly_Criteria': weekly_criteria,
                        'Daily_Criteria': daily_criteria
                    })

            processed += 1

        except Exception as e:
            st.warning(f"⚠️ Error processing {ticker}: {str(e)}")
            errors += 1

        progress_bar.progress((i + 1) / total_stocks)

    status_text.text(f"Screening selesai! Processed: {processed}, Errors: {errors}, Results: {len(results)}")
    return results

# Main app
def main():
    # Parse input saham
    stock_list = [ticker.strip() for ticker in stock_input.split('\n') if ticker.strip()]

    st.sidebar.markdown(f"**Total saham: {len(stock_list)}**")

    # Display current configuration
    with st.sidebar.expander("📊 Current Configuration"):
        st.write("**Parameters:**")
        st.write(f"- Weekly RSI Min: {weekly_rsi_min}")
        st.write(f"- Weekly ADX Min: {weekly_adx_min}")
        st.write(f"- Daily RSI Range: {daily_rsi_min}-{daily_rsi_max}")
        st.write(f"- Volume Multiplier: {volume_multiplier}x")
        st.write(f"- Market Cap Min: {market_cap_selection}")
        st.write(f"- Weekly Score Threshold: {weekly_score_threshold}")
        st.write(f"- Daily Score Threshold: {daily_score_threshold}")

    # Tombol run screening
    if st.sidebar.button("🚀 Jalankan Screening", type="primary"):
        st.markdown("### 🔍 Hasil Screening Enhanced SWING")

        with st.spinner("Sedang melakukan screening..."):
            results = run_screening(stock_list)

        if results:
            # Konversi ke DataFrame
            df_results = pd.DataFrame(results)

            # Sort berdasarkan total score
            df_results = df_results.sort_values('Total_Score', ascending=False)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Saham Dianalisis", len(stock_list))
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="success-card">', unsafe_allow_html=True)
                st.metric("Saham Lolos Screening", len(results))
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                success_rate = (len(results) / len(stock_list)) * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Success Rate", f"{success_rate:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                avg_score = df_results['Total_Score'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg Total Score", f"{avg_score:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Tabel hasil
            st.markdown("### 📊 Saham yang Lolos Screening")

            # Format tabel untuk display
            display_columns = CONFIG['export'].get('columns_to_export', [
                'Ticker', 'Price', 'Volume_Ratio', 'Daily_RSI', 
                'Weekly_RSI', 'Weekly_ADX', 'Weekly_UO', 
                'Weekly_Score', 'Daily_Score', 'Total_Score'
            ])

            # Filter columns that exist in df_results
            available_columns = [col for col in display_columns if col in df_results.columns]
            display_df = df_results[available_columns].copy()

            # Format angka dengan error handling
            numeric_columns = ['Price', 'Volume_Ratio', 'Daily_RSI', 'Weekly_RSI', 'Weekly_ADX', 'Weekly_UO']
            for col in numeric_columns:
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # Detail analisis untuk saham terpilih
            if len(results) > 0:
                st.markdown("### 🔍 Detail Analisis")
                selected_ticker = st.selectbox(
                    "Pilih saham untuk analisis detail:",
                    df_results['Ticker'].tolist()
                )

                if selected_ticker:
                    selected_data = df_results[df_results['Ticker'] == selected_ticker].iloc[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Weekly Criteria:**")
                        weekly_criteria = selected_data['Weekly_Criteria']
                        for criteria, passed in weekly_criteria.items():
                            status = "✅" if passed else "❌"
                            st.write(f"{status} {criteria.replace('_', ' ').title()}")

                    with col2:
                        st.markdown("**Daily Criteria:**")
                        daily_criteria = selected_data['Daily_Criteria']
                        for criteria, passed in daily_criteria.items():
                            status = "✅" if passed else "❌"
                            st.write(f"{status} {criteria.replace('_', ' ').title()}")

                    # Chart saham dengan config
                    st.markdown("### 📈 Chart Analisis")
                    chart_period = CONFIG['chart'].get('period', '3mo')
                    data, _ = get_stock_data(selected_ticker, period=chart_period)

                    if data is not None:
                        # Hitung indikator untuk chart
                        data['SMA5'] = ta.sma(data['Close'], length=5)
                        data['SMA20'] = ta.sma(data['Close'], length=20)
                        data['RSI'] = ta.rsi(data['Close'], length=14)

                        # Create subplots
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=('Price & Moving Averages', 'RSI'),
                            row_width=[0.7, 0.3]
                        )

                        # Get colors from config
                        colors = CONFIG['chart'].get('colors', {})

                        # Price chart
                        fig.add_trace(
                            go.Candlestick(
                                x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )

                        # Moving averages
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['SMA5'],
                                name='SMA5',
                                line=dict(color=colors.get('sma5', 'orange'), width=1)
                            ),
                            row=1, col=1
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['SMA20'],
                                name='SMA20',
                                line=dict(color=colors.get('sma20', 'red'), width=1)
                            ),
                            row=1, col=1
                        )

                        # RSI
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['RSI'],
                                name='RSI',
                                line=dict(color=colors.get('rsi', 'purple'), width=1)
                            ),
                            row=2, col=1
                        )

                        # RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)

                        chart_height = CONFIG['chart'].get('height', 600)
                        fig.update_layout(
                            title=f"{selected_ticker} - Technical Analysis",
                            xaxis_rangeslider_visible=False,
                            height=chart_height
                        )

                        st.plotly_chart(fig, use_container_width=True)

            # Download hasil dengan config filename
            st.markdown("### 💾 Download Hasil")
            filename_format = CONFIG['export'].get('filename_format', 'swing_screening_results_%Y%m%d_%H%M.csv')
            filename = datetime.now().strftime(filename_format)

            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )

        else:
            st.warning("Tidak ada saham yang memenuhi kriteria screening.")

    # Informasi algoritma
    with st.expander("ℹ️ Tentang Algoritma Enhanced SWING"):
        st.markdown("""
        **Enhanced SWING Screener** menggunakan algoritma multi-timeframe yang menggabungkan:

        **Weekly Screening (9 Kriteria):**
        - Higher highs & higher lows
        - Price above SMA20 & SMA50
        - SMA alignment (5>10>20)
        - RSI bullish (>52)
        - ADX strength (>18)
        - Ultimate Oscillator (>45)
        - MACD bullish crossover
        - Weekly bullish candle

        **Daily Screening (9 Kriteria):**
        - Price above SMA5 & SMA20
        - Price above weekly SMA5
        - RSI range (50-75)
        - MACD bullish
        - Volume surge (>1.2x avg)
        - Bullish candle pattern
        - Normal volatility
        - Strong candle body

        **Target Success Rate: 81%+**

        **Configuration:** Parameters loaded from config.py with fallback to built-in defaults.
        """)

if __name__ == "__main__":
    main()
