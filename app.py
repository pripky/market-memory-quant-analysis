import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf

# Force clear styling and disable dynamic chunks to prevent tunnel errors
st.markdown("<style>div.stSlider {opacity: 1 !important;}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Market Memory Dashboard", layout="wide")

st.title("📊 Market Memory: Dynamic Quantitative Dashboard")
st.markdown("""
    Move the sliders in the sidebar to see the statistical metrics update in real time. 
    Notice how changing timeframes or indices constantly confirms the core thesis: **Direction is a random coin-flip, but risk/magnitude persists in highly predictable cycles.**
""")

# ----------------------------------------------------
# INTERACTIVE SIDEBAR INTERFACES
# ----------------------------------------------------
st.sidebar.header("🔧 Interactive Parameters")

ticker = st.sidebar.selectbox(
    "Select Asset Index", 
    ["^GSPC", "^DJI", "^IXIC", "SPY"], 
    index=0, 
    help="^GSPC: S&P 500, ^DJI: Dow Jones, ^IXIC: Nasdaq, SPY: S&P 500 ETF"
)

# Responsive date sliders
start_year = st.sidebar.slider("Start Year", 2000, 2015, 2000)
end_year = st.sidebar.slider("End Year", 2016, 2026, 2026)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
vol_window = st.sidebar.slider("Rolling Vol Window (Days)", 5, 60, 20)
max_streak = st.sidebar.slider("Max Explored Streak Length", 3, 7, 5)
max_lag = st.sidebar.slider("Autocorrelation Lags (Days)", 5, 30, 20)

# ----------------------------------------------------
# REACTIVE DATA PROCESSING PIPELINE
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess(ticker, start, end):
    df = yf.download(ticker, start=f"{start}-01-01", end=f"{end}-12-31", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [str(col).lower() for col in df.columns]
    target_col = 'adj close' if 'adj close' in df.columns else 'close'
    
    df_clean = df[[target_col]].copy()
    df_clean.columns = ['adj_close']
    df_clean['raw_return'] = df_clean['adj_close'].pct_change()
    df_clean['abs_return'] = df_clean['raw_return'].abs()
    return df_clean.dropna()

data = load_and_preprocess(ticker, start_year, end_year)

# ----------------------------------------------------
# COMPUTE REACTIVE METRICS
# ----------------------------------------------------
# 1. Streak Probabilities
data['direction'] = (data['raw_return'] > 0).astype(int)
base_rate = data['direction'].mean()

streak_counts = []
current_streak = 0
for i in range(len(data)):
    if i == 0:
        streak_counts.append(0)
        continue
    prev_dir = data['direction'].iloc[i-1]
    if prev_dir == 1:
        current_streak = current_streak + 1 if current_streak > 0 else 1
    else:
        current_streak = current_streak - 1 if current_streak < 0 else -1
    streak_counts.append(current_streak)
data['prior_streak'] = streak_counts

streak_analysis = []
for s in range(1, max_streak + 1):
    if s == max_streak:
        subset = data[data['prior_streak'] >= s]
        label = f"{s}+ Up Days"
    else:
        subset = data[data['prior_streak'] == s]
        label = f"{s} Up Day(s)"
    
    if len(subset) > 0:
        prob = subset['direction'].mean() * 100
        streak_analysis.append({"Context": label, "Probability": prob})

streak_df = pd.DataFrame(streak_analysis)

# 2. Volatility & ACF Computation
data['rolling_vol'] = data['raw_return'].rolling(window=vol_window).std() * np.sqrt(252) * 100
acf_raw = acf(data['raw_return'], nlags=max_lag, fft=True)[1:]
acf_abs = acf(data['abs_return'], nlags=max_lag, fft=True)[1:]
conf_interval = 2 / np.sqrt(len(data))

# ----------------------------------------------------
# DYNAMIC PLOTLY VISUALIZATIONS
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Proof A: Direction Has No Memory")
    fig_streak = go.Figure()
    fig_streak.add_trace(go.Bar(
        x=streak_df['Context'], y=streak_df['Probability'],
        marker_color='rgb(50, 120, 180)', text=[f"{p:.1f}%" for p in streak_df['Probability']], textposition='auto'
    ))
    fig_streak.add_shape(type="line", x0=-0.5, x1=max_streak-0.5, y0=base_rate*100, y1=base_rate*100,
                        line=dict(color="Red", width=2, dash="dash"))
    fig_streak.update_layout(
        title=f"Next-Day Up Probability vs. Base Rate ({base_rate*100:.2f}%)",
        yaxis=dict(title="Probability (%)", range=[40, 60]),
        height=400, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_streak, use_container_width=True)

with col2:
    st.subheader("Proof B: Volatility Has Regime Memory")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data.index, y=data['rolling_vol'], line=dict(color='firebrick', width=1.2)))
    fig_vol.update_layout(
        title=f"{vol_window}-Day Rolling Annualized Volatility Time-Series",
        yaxis_title="Annualized Volatility (%)",
        height=400, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")
st.subheader("🔄 Statistical Face-off: Autocorrelation Profiles")

fig_acf = make_subplots(rows=1, cols=2, subplot_titles=(
    "Direction Memory (Raw Signed Returns Correlation)", 
    "Magnitude Memory (Absolute Returns Correlation)"
))

lags = list(range(1, max_lag + 1))
fig_acf.add_trace(go.Bar(x=lags, y=acf_raw, marker_color='darkgrey'), row=1, col=1)
fig_acf.add_trace(go.Bar(x=lags, y=acf_abs, marker_color='seagreen'), row=1, col=2)

for col_idx in [1, 2]:
    fig_acf.add_shape(type="line", x0=0.5, x1=max_lag+0.5, y0=conf_interval, y1=conf_interval, line=dict(color="blue", dash="dash", width=1), row=1, col=col_idx)
    fig_acf.add_shape(type="line", x0=0.5, x1=max_lag+0.5, y0=-conf_interval, y1=-conf_interval, line=dict(color="blue", dash="dash", width=1), row=1, col=col_idx)

fig_acf.update_layout(height=400, showlegend=False, yaxis=dict(range=[-0.15, 0.35]), yaxis2=dict(range=[-0.15, 0.35]))
st.plotly_chart(fig_acf, use_container_width=True)

# Interactive KPIs at the footer
st.markdown("### 📊 Active Model Summary Metrics")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Explored Horizon Data Size", f"{len(data):,} Trading Days")
kpi2.metric(f"Lag-1 Direction ACF", f"{acf_raw[0]:.4f}")
kpi3.metric(f"Lag-1 Volatility ACF", f"{acf_abs[0]:.4f}")
