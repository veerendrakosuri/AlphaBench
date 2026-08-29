from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AlphaBench", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


st.title("AlphaBench")
st.caption("Walk-forward equity return forecasting & backtesting")

st.warning(
    "**Educational research artifact — not investment advice.** All results are "
    "historical simulations with no guarantee of future performance. Do not trade "
    "on these outputs.",
    icon="⚠️",
)

try:
    health = get("/health")
except Exception as exc:
    st.error(f"API unreachable at {API}: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Model", "loaded" if health["model_loaded"] else "missing")
c2.metric("Symbols", health["n_symbols"])
c3.metric("Data through", health["data_last_updated"] or "—")

symbols = get("/tickers")["tickers"]
with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Ticker", symbols)
    use_default_threshold = st.checkbox(
        "Use configured default threshold",
        value=True,
        help="The default is calibrated to this model's actual probability range. "
        "Uncheck to override it manually.",
    )
    threshold = st.slider(
        "Probability threshold (override)",
        0.45,
        0.60,
        0.5164,
        0.005,
        disabled=use_default_threshold,
    )
    st.caption("Higher threshold → fewer, more selective trades.")

tab1, tab2, tab3 = st.tabs(["Signal", "Backtest", "Validation"])

with tab1:
    p = get(f"/predict/{symbol}")
    a, b, c = st.columns(3)
    a.metric("P(up)", f"{p['probability_up']:.3f}")
    b.metric("Signal", p["signal"])
    c.metric("As of", p["as_of"])
    st.progress(p["probability_up"])
    st.caption(
        "A probability near 0.50 means the model sees no edge. That is the "
        "expected state on most days and is a feature, not a failure."
    )

with tab2:
    bt_kwargs = {"symbol": symbol}
    if not use_default_threshold:
        bt_kwargs["threshold"] = threshold
    bt = get("/backtest", **bt_kwargs)
    m = bt["metrics"]
    eq = pd.DataFrame(bt["equity_curve"])
    eq["date"] = pd.to_datetime(eq["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"], name="Strategy (net of costs)"))
    fig.add_trace(
        go.Scatter(x=eq["date"], y=eq["benchmark"], name="Buy & hold", line=dict(dash="dash"))
    )
    fig.update_layout(height=440, yaxis_title="Growth of 1 unit", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    k = st.columns(4)
    k[0].metric("Sharpe (net)", f"{m['sharpe']:.3f}", delta=f"{m['excess_sharpe']:+.3f} vs B&H")
    k[1].metric("Ann. return", f"{m['ann_return']:.2%}")
    k[2].metric("Max drawdown", f"{m['max_drawdown']:.2%}")
    k[3].metric("Hit rate", f"{m['hit_rate']:.2%}")

    with st.expander("All metrics"):
        st.json(m)

with tab3:
    mt = get("/metrics")
    wf = pd.DataFrame(mt["walkforward"])
    st.subheader("Walk-forward results by fold")
    st.dataframe(wf, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=wf["val_year"], y=wf["auc"], name="AUC"))
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="chance (0.50)")
    fig.update_layout(height=340, yaxis_title="ROC-AUC", yaxis_range=[0.45, 0.65])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Mean AUC {wf['auc'].mean():.4f} across {len(wf)} folds. "
        "Values of 0.52–0.55 are realistic for daily equity direction. "  # noqa: RUF001
        "Anything above 0.60 should trigger a leakage audit before it is believed."
    )
