# dashboard.py
"""
Real-Time Trading Dashboard with Live Updates
Similar to actual trading platforms
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime
import numpy as np
import config

st.set_page_config(
    page_title="Real-Time Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .profit {color: #00ff00; font-weight: bold;}
    .loss {color: #ff0000; font-weight: bold;}
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


def load_portfolio_data():
    """Load real-time portfolio data"""
    portfolio_file = os.path.join(config.RESULTS_DIR, 'realtime_portfolio.csv')
    
    if os.path.exists(portfolio_file):
        df = pd.read_csv(portfolio_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()


def load_trades_data():
    """Load real-time trades data"""
    trades_file = os.path.join(config.RESULTS_DIR, 'realtime_trades.csv')
    
    if os.path.exists(trades_file):
        df = pd.read_csv(trades_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()


def calculate_metrics(portfolio_df, trades_df):
    """Calculate trading metrics"""
    if portfolio_df.empty:
        return {}
    
    initial_capital = config.INITIAL_CAPITAL
    current_value = portfolio_df['total_value'].iloc[-1]
    total_pnl = current_value - initial_capital
    total_return = (total_pnl / initial_capital) * 100
    
    # Calculate max drawdown
    peak = portfolio_df['total_value'].expanding().max()
    drawdown = (portfolio_df['total_value'] - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Calculate returns for Sharpe
    returns = portfolio_df['total_value'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252 * 24) * returns.mean() / (returns.std() + 1e-9)
    
    # Trade statistics
    total_trades = len(trades_df)
    if total_trades > 0:
        completed_trades = trades_df[trades_df['action'] == 'SELL']
        if not completed_trades.empty:
            winning_trades = completed_trades[completed_trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(completed_trades) * 100
            
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            losing_trades = completed_trades[completed_trades['pnl'] <= 0]
            avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
            
            profit_factor = abs(winning_trades['pnl'].sum() / (losing_trades['pnl'].sum() - 1e-9)) if not losing_trades.empty else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return {
        'initial_capital': initial_capital,
        'current_value': current_value,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def create_portfolio_chart(portfolio_df):
    """Create portfolio value chart"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Portfolio Value', 'Profit/Loss'),
        vertical_spacing=0.1
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add initial capital line
    fig.add_hline(
        y=config.INITIAL_CAPITAL,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        row=1, col=1
    )
    
    # P&L
    fig.add_trace(
        go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['total_pnl'],
            mode='lines',
            name='P&L',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)'
    )
    
    return fig


def create_trades_chart(trades_df):
    """Create trades visualization"""
    if trades_df.empty:
        return None
    
    fig = go.Figure()
    
    # Buy trades
    buys = trades_df[trades_df['action'] == 'BUY']
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['timestamp'],
            y=buys['price'],
            mode='markers',
            name='Buy',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='#00ff00',
                line=dict(width=2, color='white')
            ),
            text=[f"{row['symbol']}<br>{row['shares']:.2f} shares" for _, row in buys.iterrows()],
            hovertemplate='<b>BUY</b><br>%{text}<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
        ))
    
    # Sell trades
    sells = trades_df[trades_df['action'] == 'SELL']
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['timestamp'],
            y=sells['price'],
            mode='markers',
            name='Sell',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='#ff0000',
                line=dict(width=2, color='white')
            ),
            text=[f"{row['symbol']}<br>{row['shares']:.2f} shares<br>P&L: ${row.get('pnl', 0):.2f}" 
                  for _, row in sells.iterrows()],
            hovertemplate='<b>SELL</b><br>%{text}<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Trade Execution',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        height=400,
        hovermode='closest',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)'
    )
    
    return fig


def create_pnl_distribution(trades_df):
    """Create P&L distribution chart"""
    completed = trades_df[trades_df['action'] == 'SELL']
    
    if completed.empty or 'pnl' not in completed.columns:
        return None
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=completed['pnl'],
        nbinsx=20,
        marker=dict(
            color=completed['pnl'],
            colorscale=[[0, '#ff0000'], [0.5, '#ffff00'], [1, '#00ff00']],
            line=dict(color='white', width=1)
        ),
        name='P&L Distribution'
    ))
    
    fig.update_layout(
        title='Profit/Loss Distribution',
        xaxis_title='P&L ($)',
        yaxis_title='Frequency',
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        showlegend=False
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Real-Time Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 60, 5)
        
        st.markdown("---")
        st.header("📊 Status")
        
        # Check if trading is active
        portfolio_df = load_portfolio_data()
        if not portfolio_df.empty:
            last_update = portfolio_df['timestamp'].iloc[-1]
            time_since = (datetime.now() - last_update).total_seconds()
            
            if time_since < 120:  # Active if updated in last 2 minutes
                st.success("🟢 Trading Active")
            else:
                st.warning("🟡 Trading Paused")
            
            st.info(f"Last update: {last_update.strftime('%H:%M:%S')}")
        else:
            st.error("🔴 No Data")
            st.info("Start trading to see live data")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    # Load data
    portfolio_df = load_portfolio_data()
    trades_df = load_trades_data()
    
    if portfolio_df.empty:
        st.warning("⚠️ No trading data available yet. Start trading to see live updates!")
        st.info("Run: `python run.py --mode trade`")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, trades_df)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${metrics['current_value']:,.2f}",
            f"{metrics['total_return']:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Total P&L",
            f"${metrics['total_pnl']:,.2f}",
            delta_color="normal" if metrics['total_pnl'] >= 0 else "inverse"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            "Risk-adjusted"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2f}%",
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            "Total Trades",
            f"{metrics['total_trades']}",
            f"WR: {metrics['win_rate']:.1f}%"
        )
    
    st.markdown("---")
    
    # Main chart
    st.subheader("📊 Portfolio Performance")
    portfolio_chart = create_portfolio_chart(portfolio_df)
    st.plotly_chart(portfolio_chart, use_container_width=True)
    
    # Two columns for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Trade Execution")
        if not trades_df.empty:
            trades_chart = create_trades_chart(trades_df)
            if trades_chart:
                st.plotly_chart(trades_chart, use_container_width=True)
        else:
            st.info("No trades executed yet")
    
    with col2:
        st.subheader("📈 P&L Distribution")
        pnl_chart = create_pnl_distribution(trades_df)
        if pnl_chart:
            st.plotly_chart(pnl_chart, use_container_width=True)
        else:
            st.info("No completed trades yet")
    
    st.markdown("---")
    
    # Detailed metrics
    st.subheader("📋 Detailed Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 💰 Performance")
        st.write(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        st.write(f"Current Value: ${metrics['current_value']:,.2f}")
        
        pnl_class = "profit" if metrics['total_pnl'] >= 0 else "loss"
        st.markdown(f"Total P&L: <span class='{pnl_class}'>${metrics['total_pnl']:,.2f}</span>", 
                   unsafe_allow_html=True)
        st.markdown(f"Total Return: <span class='{pnl_class}'>{metrics['total_return']:+.2f}%</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("##### 📊 Risk Metrics")
        st.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        st.write(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Calculate volatility
        returns = portfolio_df['total_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 * 24) * 100
        st.write(f"Volatility: {volatility:.2f}%")
    
    with col3:
        st.markdown("##### 🎯 Trading Stats")
        st.write(f"Total Trades: {metrics['total_trades']}")
        st.write(f"Win Rate: {metrics['win_rate']:.1f}%")
        st.write(f"Avg Win: ${metrics['avg_win']:.2f}")
        st.write(f"Avg Loss: ${metrics['avg_loss']:.2f}")
        st.write(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    st.markdown("---")
    
    # Recent trades table
    st.subheader("📝 Recent Trades")
    
    if not trades_df.empty:
        # Show last 10 trades
        recent_trades = trades_df.tail(10).sort_values('timestamp', ascending=False)
        
        # Format for display
        display_df = recent_trades[['timestamp', 'symbol', 'action', 'shares', 'price']].copy()
        
        if 'pnl' in recent_trades.columns:
            display_df['pnl'] = recent_trades['pnl'].fillna(0)
            display_df['pnl_pct'] = recent_trades['pnl_pct'].fillna(0)
        
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        display_df['shares'] = display_df['shares'].round(4)
        display_df['price'] = display_df['price'].round(2)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No trades yet")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()