import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/stock_data.csv")

    df["date"] = pd.to_datetime(df["date"])

    return df


data = load_data()


# ==========================================
# 3. SIDEBAR
# ==========================================

st.sidebar.title("📈 Stock Prediction")

st.sidebar.write("Machine Learning Dashboard")

companies = sorted(data["Name"].dropna().unique())

selected_stock = st.sidebar.selectbox(
    "Select Company",
    companies
)

st.sidebar.divider()

st.sidebar.info(
    "This application uses Linear Regression "
    "to estimate the next day's closing price."
)


# ==========================================
# 4. PREPARE SELECTED STOCK DATA
# ==========================================

stock_data = data[
    data["Name"] == selected_stock
].copy()

stock_data = stock_data.sort_values("date")

features = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

stock_data["Next_Close"] = stock_data["close"].shift(-1)

stock_data["Target_Date"] = stock_data["date"].shift(-1)

stock_data = stock_data.dropna().reset_index(drop=True)


if len(stock_data) < 10:
    st.error("Not enough records for this company.")
    st.stop()


# ==========================================
# 5. TRAIN AND TEST MODEL
# ==========================================

split = int(len(stock_data) * 0.8)

X = stock_data[features]
y = stock_data["Next_Close"]

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ==========================================
# 6. MODEL PERFORMANCE
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


# ==========================================
# 7. DASHBOARD TITLE
# ==========================================

st.title("📈 Stock Price Prediction System")

st.write(
    "A Machine Learning dashboard for historical stock "
    "analysis and next-day closing price estimation."
)

st.caption(f"Selected Company: {selected_stock}")

st.divider()


# ==========================================
# 8. STOCK SUMMARY
# ==========================================

latest = data[
    data["Name"] == selected_stock
].sort_values("date").iloc[-1]

total_records = len(
    data[data["Name"] == selected_stock]
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{total_records:,}"
    )

with col2:
    st.metric(
        "Latest Closing Price",
        f"${latest['close']:.2f}"
    )

with col3:
    st.metric(
        "Average Closing Price",
        f"${stock_data['close'].mean():.2f}"
    )

with col4:
    st.metric(
        "Model R² Score",
        f"{r2:.3f}"
    )


# ==========================================
# 9. DATE FILTER
# ==========================================

st.divider()

st.header("📅 Filter Historical Data")

minimum_date = stock_data["date"].min().date()
maximum_date = stock_data["date"].max().date()

selected_dates = st.date_input(
    "Select date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date
)


# ==========================================
# 10. FILTER DATA
# ==========================================

if len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_data = stock_data[
        (stock_data["date"].dt.date >= start_date)
        & (stock_data["date"].dt.date <= end_date)
    ].copy()

else:

    filtered_data = stock_data.copy()


if filtered_data.empty:

    st.warning("No records found for the selected date range.")

else:

    # ======================================
    # 11. HISTORICAL STOCK PRICE GRAPH
    # ======================================

    st.divider()

    st.header("📈 Historical Stock Prices")

    history_chart = filtered_data[
        ["date", "close"]
    ].copy()

    history_chart = history_chart.set_index("date")

    history_chart.columns = ["Closing Price"]

    st.line_chart(history_chart)

    st.caption(
        "This graph displays historical closing prices "
        "for the selected company and date range."
    )


# ==========================================
# 12. ACTUAL VS PREDICTED GRAPH
# ==========================================

st.divider()

st.header("📊 Actual vs Predicted Prices")

comparison = pd.DataFrame({
    "Date": stock_data["Target_Date"].iloc[split:].to_numpy(),
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})

comparison["Date"] = pd.to_datetime(comparison["Date"])

if len(selected_dates) == 2:

    start_date, end_date = selected_dates

    comparison = comparison[
        (comparison["Date"].dt.date >= start_date)
        & (comparison["Date"].dt.date <= end_date)
    ]

if comparison.empty:

    st.info(
        "No test predictions are available for this date range. "
        "Try selecting a wider date range."
    )

else:

    chart_data = comparison.set_index("Date")[
        ["Actual Price", "Predicted Price"]
    ]

    st.line_chart(chart_data)

    st.caption(
        "Actual and predicted prices are compared on the "
        "held-out test data. This is a historical evaluation, "
        "not a forecast of guaranteed future prices."
    )


# ==========================================
# 13. DOWNLOAD REPORT
# ==========================================

st.divider()

st.header("📥 Download Prediction Report")

download_data = comparison.copy()

if not download_data.empty:

    csv_data = download_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv_data,
        file_name=f"{selected_stock}_prediction_report.csv",
        mime="text/csv"
    )

    st.write(
        "The report contains the date, actual closing price, "
        "and predicted closing price for the test period."
    )

else:

    st.info(
        "Choose a wider date range to download prediction records."
    )


# ==========================================
# 14. MODEL PERFORMANCE DETAILS
# ==========================================

st.divider()

st.header("🎯 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Mean Absolute Error",
        f"{mae:.4f}"
    )

with col2:

    st.metric(
        "Root Mean Squared Error",
        f"{rmse:.4f}"
    )

with col3:

    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

with st.expander("Understand the metrics"):

    st.write(
        "**MAE:** Average absolute difference between "
        "actual and predicted prices."
    )

    st.write(
        "**RMSE:** Measures prediction error while giving "
        "larger errors more weight."
    )

    st.write(
        "**R² Score:** Indicates how well the model explains "
        "variation in the test data. It can be negative."
    )


# ==========================================
# 15. MANUAL PRICE PREDICTION
# ==========================================

st.divider()

st.header("🔮 Predict Next Day Closing Price")

st.write(
    f"Enter trading details for {selected_stock}."
)

col1, col2 = st.columns(2)

with col1:

    open_price = st.number_input(
        "Open Price",
        min_value=0.0,
        value=float(latest["open"]),
        step=0.01
    )

    high_price = st.number_input(
        "High Price",
        min_value=0.0,
        value=float(latest["high"]),
        step=0.01
    )

    low_price = st.number_input(
        "Low Price",
        min_value=0.0,
        value=float(latest["low"]),
        step=0.01
    )

with col2:

    close_price = st.number_input(
        "Closing Price",
        min_value=0.0,
        value=float(latest["close"]),
        step=0.01
    )

    volume = st.number_input(
        "Trading Volume",
        min_value=0,
        value=int(latest["volume"]),
        step=1000
    )


if st.button("🔮 Predict Next Day Price", type="primary"):

    if high_price < low_price:

        st.error(
            "High Price cannot be lower than Low Price."
        )

    elif not (
        low_price <= open_price <= high_price
        and low_price <= close_price <= high_price
    ):

        st.error(
            "Open Price and Closing Price must be between "
            "Low Price and High Price."
        )

    else:

        input_data = pd.DataFrame(
            [[
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ]],
            columns=features
        )

        predicted_price = model.predict(input_data)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            f"{selected_stock} Predicted Closing Price",
            f"${predicted_price:.2f}"
        )


# ==========================================
# 16. DISCLAIMER
# ==========================================

st.divider()

st.caption(
    "Educational project only. Stock prices are uncertain. "
    "Predictions are estimates and are not financial advice."
)