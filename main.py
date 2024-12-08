import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf #用於下載股票數據
from prophet import Prophet #進行時間序列預測
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2020-01-01" #開始日期
TODAY = date.today().strftime("%Y-%m-%d") #當天日期

st.title("Stock Prediction App")

#從下拉選單選取股票代號
stocks = ("AAPL","GOOG","TSM")
selected_stock = st.selectbox("Select dataset for prediction",stocks)
# 預測年限，最少1年，最多4年
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365 #預測總天數

# 定義一個緩存函數，用來載入數據
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)#下載指定股票數據
    data.reset_index(inplace = True)#重置索引(原本的索引為日期=>改成0,1,2...)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")
data['Date'] = pd.to_datetime(data['Date']).dt.date
# 合併多重索引欄位名稱（若存在多重索引）
data.columns = data.columns.get_level_values(0)# 保留欄位名稱的第一層索引
st.subheader('Raw data')
st.write(data.tail())# 顯示數據的最後幾行
# 定義函數來繪製原始數據的圖表
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stack_open')) # 添加開盤價的折線圖
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stack_close'))# 添加收盤價的折線圖
    fig.layout.update(title_text="Time Series Date",xaxis_rangeslider_visible=True) #設置範圍滑塊
    st.plotly_chart(fig)
plot_raw_data()

# Forecasting
df_train = data[['Date','Close']]#將收盤價用作訓練資料
df_train = df_train.rename(columns={"Date":"ds","Close":"y"}) # 將日期和收盤價的欄位重命名以符合 Prophet 的格式
m = Prophet()
m.fit(df_train)# train
future = m.make_future_dataframe(periods=period)# 生成指定天數的未來日期
forecast = m.predict(future)# 預測未來的數據

st.subheader('Forecast data')
st.write(forecast.tail())

# 使用 Prophet 和 Plotly 繪製預測圖
st.write('forecast data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

# 顯示預測結果的組成部分
st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)  