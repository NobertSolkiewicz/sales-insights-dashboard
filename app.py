import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Sales Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/sales.csv", encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df

def calculate_kpis(data):
    total_sales = data["Sales"].sum()
    total_profit = data["Profit"].sum()
    total_orders = data["Order ID"].nunique()

    if total_orders > 0:
        avg_order_value = total_sales / total_orders
    else:
        avg_order_value = 0

    return total_sales, total_profit, total_orders, avg_order_value

def create_category_sales_chart(data):
    sales_by_category = data.groupby("Category")["Sales"].sum().sort_values(
        ascending=False).to_frame().reset_index()

    fig = px.bar(
        sales_by_category,
        x="Category",
        y="Sales",
    )
    return fig

def create_region_profit_chart(data):
    profit_by_region = data.groupby("Region")["Profit"].sum().sort_values(ascending=False).to_frame().reset_index()
    fig = px.bar(
        profit_by_region,
        x="Region",
        y="Profit",
    )
    return fig

def create_trend_chart(data, selected_metric):
    months = data["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly_metric = data.groupby(months)[selected_metric].sum().to_frame().reset_index()

    fig = px.line(
        monthly_metric,
        x="Order Date",
        y=selected_metric,
        height=600,
        markers=True
    )

    return fig

def create_top_products_chart(data):
    sales_by_product = (
        data.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index())

    fig = px.bar(
        sales_by_product,
        x="Sales",
        y="Product Name",
        height=500,
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    return fig

def create_scatter_plot(data, column):
    fig_scatter = px.scatter(
        data,
        x="Sales",
        y="Profit",
        color=column,
        hover_data=["Product Name", "Region"],
        opacity=0.7,
        height=650,
    )

    fig_scatter.update_traces(
        marker=dict(
            size=6,
            line=dict(width=1, color="white")
        )
    )
    fig_scatter.add_hline(y=0,
                          line_dash="dash",
                          line_color="red"
    )

    return fig_scatter

# Data loading
df = load_data()

st.title("📊 Sales Insights Dashboard")
st.write("Explore sales performance across regions, categories, and time.")

# Sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.selectbox("Select Region", ["All"] + sorted(df["Region"].dropna().unique().tolist()))

selected_category = st.sidebar.selectbox("Select Category", ["All"] + sorted(df["Category"].dropna().unique().tolist()))

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

data_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date)

filtered_df = df.copy()

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

start_date, end_date = data_range

filtered_df = filtered_df[
    (filtered_df["Order Date"].dt.date >= start_date) &
    (filtered_df["Order Date"].dt.date <= end_date)]

# Dashboard overview
st.header("Overview")

total_sales, total_profit, total_orders, avg_order_value = calculate_kpis(filtered_df)

kpi_columns = st.columns(4)
kpi_columns[0].metric("Total sales", f"${total_sales:,.2f}")
kpi_columns[1].metric("Total profit", f"${total_profit:,.2f}")
kpi_columns[2].metric("Total orders", total_orders)
kpi_columns[3].metric("Average order value", f"${avg_order_value:,.2f}")

# Business insights
st.header("Business Insights")

fig_category = create_category_sales_chart(filtered_df)

fig_region = create_region_profit_chart(filtered_df)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Sales by category")
    st.plotly_chart(fig_category)

with chart_col2:
    st.subheader("Profit by region")
    st.plotly_chart(fig_region)

# Trend analysis
st.header("Trend Analysis")

selected_metric = st.selectbox("Select Metric: ", ["Sales", "Profit", "Quantity"])
fig_month = create_trend_chart(filtered_df, selected_metric)

st.subheader("Sales over time")
st.plotly_chart(fig_month)

# Data science insights
st.header("Product Insights")

fig_product = create_top_products_chart(filtered_df)

st.subheader("Top 10 Best-Selling Products")
st.plotly_chart(fig_product)

show_ds_chart = st.checkbox("Show Advanced Analysis")

if show_ds_chart:
    st.header("Data Science Insights")
    st.subheader("Relationship Between Sales and Profit")

    profitable = filtered_df["Profit"] >= 0
    loss_making = filtered_df["Profit"] < 0
    scatter_filter = st.selectbox("Select Profit View", ["All", "Profitable only", "Loss-making only"])
    scatter_df = filtered_df.copy()

    if scatter_filter == "Profitable only":
        scatter_df = scatter_df[profitable]
    elif scatter_filter == "Loss-making only":
        scatter_df = scatter_df[loss_making]

    selected_color = st.selectbox("Select Color Dimension", ["Category", "Region", "Segment"])
    fig_scatter = create_scatter_plot(scatter_df, selected_color)
    st.write(f"Showing {scatter_df.shape[0]} records")
    st.plotly_chart(fig_scatter)

show_raw_data = st.checkbox("Show Filtered Data Table")
if show_raw_data:
    st.header("Filtered Data Preview")
    row_numbers = st.selectbox("Select Number of Rows: ", [5, 10, 20 ,50])
    st.dataframe(filtered_df.head(row_numbers))
    st.write(f"Showing {filtered_df.shape[0]} records after applying filters")

#Machine learning section
feature_columns = ["Sales", "Quantity", "Discount", "Category", "Region", "Sub-Category", "Segment"]
X = df[feature_columns]
y = df["Profit"]
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded,
        y,
        test_size=0.2,
         random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.header("Model Performance")

metric_col1_tree, metric_col2_tree = st.columns(2)
metric_col1_tree.metric("MAE", f"{mae:.2f}")
metric_col2_tree.metric("R²", f"{r2:.2f}")

st.header("Prediction Inputs")

input_col1, input_col2 = st.columns(2)
with input_col1:
    sales_input = st.number_input("Sales",
        value=0.0,
        step=1.0
    )
    quantity_input = st.number_input(
        "Quantity",
        value=0,
        step=1
    )
    discount_input = st.number_input("Discount",
        value=0.0,
         min_value=0.0,
         max_value=1.0,
         step=0.05
    )

with input_col2:
    category_input = st.selectbox("Select Category",
        sorted(
            df["Category"]
            .dropna()
            .unique()
            .tolist()))
    region_input = st.selectbox("Select Region",
        sorted(
            df["Region"]
            .dropna()
            .unique()
            .tolist()))
    sub_category_input = st.selectbox(
        "Select Sub-Category",
        sorted(df["Sub-Category"]
            .dropna()
            .unique()
            .tolist()))
    segment_input = st.selectbox(
        "Select Segment",
        sorted(df["Segment"]
               .dropna()
               .unique()
               .tolist()))

user_input_df = pd.DataFrame([
    {"Sales": sales_input,
     "Quantity": quantity_input,
     "Discount": discount_input,
     "Category": category_input,
     "Region": region_input,
     "Sub-Category": sub_category_input,
     "Segment": segment_input}
])

user_input_encoded = pd.get_dummies(user_input_df)
user_input_encoded = user_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

predicted_profit = model.predict(user_input_encoded)
st.header("Profit Prediction")
st.metric("Predicted Profit", f"{predicted_profit[0]:.2f}")