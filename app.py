import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import get_xgb_features
from src.split import time_split_3way
from src.train import train_model
from src.evaluate import evaluate
from src.backtest import backtest_trader_mode, optimize_trader_mode

st.set_page_config(page_title="ML Trading Strategy Engine", layout="wide")

st.title("📈 ML-Based Trading Strategy Backtest Engine")
st.markdown("---")

# ============= SIDEBAR CONTROLS =============
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker = st.text_input("Stock Ticker", value="AAPL", max_chars=5).upper()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date
        )
    
    st.subheader("Model Parameters")
    n_estimators = st.slider("Number of Trees", 50, 500, 300, 50)
    max_depth = st.slider("Max Depth", 5, 30, 12, 1)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 8, 1)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 4, 1)
    
    st.subheader("Backtest Parameters")
    fee_per_side = st.number_input("Fee Per Side", 0.0001, 0.01, 0.0005, 0.0001)
    
    run_button = st.button("🚀 Run Strategy", use_container_width=True)

# ============= MAIN CONTENT =============
if run_button:
    try:
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ---------- DATA LOADING ----------
        status_text.text("📊 Loading stock data...")
        progress_bar.progress(10)
        
        df_ohlcv = fetch_stock_data(
            ticker, 
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        df_ohlcv = validate_ohlcv(df_ohlcv)
        
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(20)
        
        df = get_xgb_features(df_ohlcv)
        df = df.dropna()
        
        st.success(f"✅ Loaded {len(df)} data points from {df.index[0].date()} to {df.index[-1].date()}")
        
        # ---------- DATA SPLIT ----------
        status_text.text("✂️ Splitting data (Train/Val/Test)...")
        progress_bar.progress(30)
        
        X_train, X_val, X_test, y_train, y_val, y_test = time_split_3way(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train Set", f"{len(X_train)} samples")
        with col2:
            st.metric("Val Set", f"{len(X_val)} samples")
        with col3:
            st.metric("Test Set", f"{len(X_test)} samples")
        
        # ---------- MODEL TRAINING ----------
        status_text.text("🤖 Training Random Forest model...")
        progress_bar.progress(50)
        
        model = train_model(
            X_train, y_train,
            X_val, y_val,
            model_params={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            }
        )
        
        st.success("✅ Model training complete")
        
        # ---------- PREDICTIONS ----------
        status_text.text("🎯 Generating predictions...")
        progress_bar.progress(60)
        
        X_val_scaled = model.scaler.transform(X_val)
        X_test_scaled = model.scaler.transform(X_test)
        
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # ---------- STRATEGY OPTIMIZATION ----------
        status_text.text("⚡ Optimizing trading rules (Validation)...")
        progress_bar.progress(70)
        
        opt_results = optimize_trader_mode(
            df=df_ohlcv.loc[X_val.index],
            proba=val_proba,
            fee_per_side=fee_per_side
        )
        
        if opt_results.empty:
            st.error("❌ No valid trading strategy found. Try different parameters.")
            st.stop()
        
        best = opt_results.iloc[0]
        buy_t = best["buy_threshold"]
        sell_t = best["sell_threshold"]
        hold_days = int(best["max_hold_days"])
        
        st.success("✅ Strategy optimization complete")
        
        # Show top 5 strategies
        with st.expander("📊 Top 5 Trading Strategies (Sorted by Return)"):
            st.dataframe(opt_results.head(5), use_container_width=True)
        
        # ---------- FINAL BACKTEST ----------
        status_text.text("🧪 Running final backtest (Test Set)...")
        progress_bar.progress(85)
        
        final_results, final_df = backtest_trader_mode(
            df=df_ohlcv.loc[X_test.index],
            proba=test_proba,
            buy_threshold=buy_t,
            sell_threshold=sell_t,
            max_hold_days=hold_days,
            fee_per_side=fee_per_side
        )
        
        status_text.text("📈 Generating visualizations...")
        progress_bar.progress(95)
        
        st.markdown("---")
        st.subheader("🎯 Selected Trading Strategy")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Buy Threshold", f"{buy_t:.2f}")
        with col2:
            st.metric("Sell Threshold", f"{sell_t:.2f}")
        with col3:
            st.metric("Max Hold Days", hold_days)
        with col4:
            st.metric("Fee Per Side", f"{fee_per_side:.4f}")
        
        # ---------- RESULTS ==========
        st.markdown("---")
        st.subheader("📊 Backtest Results (Test Set)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Strategy Return",
                f"{final_results['strategy_total_return']*100:.2f}%",
                delta=f"{(final_results['strategy_total_return'] - final_results['buyhold_total_return'])*100:.2f}%"
            )
        with col2:
            st.metric("Buy & Hold Return", f"{final_results['buyhold_total_return']*100:.2f}%")
        with col3:
            st.metric("Max Drawdown", f"{final_results['max_drawdown']*100:.2f}%")
        with col4:
            st.metric("Win Rate", f"{final_results['win_rate']*100:.1f}%")
        with col5:
            st.metric("Total Trades", int(final_results['trades']))
        
        # ---------- EQUITY CURVE ----------
        st.markdown("---")
        st.subheader("📈 Equity Curve")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(final_df.index, final_df["equity_strategy"], label="Strategy", linewidth=2, color="blue")
        ax.plot(final_df.index, final_df["equity_buyhold"], label="Buy & Hold", linestyle="--", linewidth=2, color="orange")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity Value")
        ax.set_title("Strategy Performance vs Buy & Hold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ---------- DRAWDOWN CHART ----------
        st.markdown("---")
        st.subheader("📉 Drawdown Analysis")
        
        peak = final_df["equity_strategy"].cummax()
        drawdown = (final_df["equity_strategy"] / peak) - 1
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(final_df.index, drawdown, 0, alpha=0.6, color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Strategy Drawdown Over Time")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ---------- CLASSIFICATION METRICS ----------
        st.markdown("---")
        st.subheader("📌 Classification Metrics (Reference)")
        
        y_pred = (test_proba >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Downtrend", "Uptrend"])
        ax.set_yticklabels(["Downtrend", "Uptrend"])
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix")
        
        # Add values to cells
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=14)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ---------- TOMORROW PREDICTION ----------
        st.markdown("---")
        st.subheader("🔮 Tomorrow Prediction")
        
        # Get latest data for prediction
        latest_features = df.iloc[-1:].drop(columns=['target'])
        latest_features_scaled = model.scaler.transform(latest_features)
        tomorrow_proba = model.predict_proba(latest_features_scaled)[0, 1]
        
        # Calculate expected move and range from recent volatility
        recent_returns = df_ohlcv['close'].pct_change().tail(20)
        avg_return = recent_returns.mean()
        volatility = recent_returns.std()
        
        # Generate prediction metrics
        direction = "UP" if tomorrow_proba >= 0.5 else "DOWN"
        confidence_pct = int(tomorrow_proba * 100) if tomorrow_proba >= 0.5 else int((1 - tomorrow_proba) * 100)
        expected_move = avg_return * 100
        range_low = (avg_return - volatility) * 100
        range_high = (avg_return + volatility) * 100
        
        # Confidence level
        if confidence_pct >= 70:
            confidence_level = "High"
        elif confidence_pct >= 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Display prediction
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Direction", f"{direction}")
        with col2:
            st.metric("Confidence", f"{confidence_pct}%")
        with col3:
            st.metric("Expected Move", f"{expected_move:+.1f}%")
        with col4:
            st.metric("Confidence Level", confidence_level)
        
        # Range display
        st.info(f"**Predicted Range:** {range_low:+.1f}% to {range_high:+.1f}%")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Strategy** to start the analysis")

