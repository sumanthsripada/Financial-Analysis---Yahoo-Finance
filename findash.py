# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:17:05 2021

@author: ssripada1
"""

    

# -*- coding: utf-8 -*-
###############################################################################
# Yahoo Finance Dashboard - Sumanth Sripada
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import streamlit as st
import yfinance as yf 
from plotly import graph_objs as go
import pandas_datareader.data as web
import datetime as dt
import numpy as np
from annotated_text import annotated_text
from plotly.subplots import make_subplots
#import plotly.express as px

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.youtube.com/channel/UCEAZeUIeJs0IjQiqTCdVSIg" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/YahooFinance?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)
side_bar = """
  <style>
    /* The whole sidebar */
    .css-1lcbmhc.e1fqkh3o0{
      margin-top: 3.8rem;
    }
     
     /* The display arrow */
    .css-sg054d.e1fqkh3o3 {
      margin-top: 5rem;
      }
  </style> 
  """
st.markdown(side_bar, unsafe_allow_html=True)
# =============================================================================

#==============================================================================
# Tab 1 -  Stock Summary 
#==============================================================================

def tab1():
    
    annotated_text(("Yahoo","Stock Summary","#3498DB"))
    col1,col2 = st.columns([2,2])
    # Add table to show stock data
    
    select_Period = ['-','1mo', '3mo','6mo','ytd','1y','2y','5y','max']
    default  = select_Period.index('1y')
    select_Period =  st.selectbox('Select Period', select_Period,index = default)
    
    @st.cache
    def GetSummary(tickers):
        return si.get_quote_table(ticker,dict_result = False)
    @st.cache
    def GetStockData(tickers, start_date, end_date):
        return pd.concat([si.get_data(tick, start_date, end_date) for tick in tickers])
        
    if ticker != '-':   
          data = yf.download(ticker, period = select_Period)
          

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.07, subplot_titles=('Stock Trend', 'Volume'), 
               row_width=[0.2, 0.7])
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'],name="Stock Trend",showlegend=True,fill='tozeroy'),row= 1,col = 1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True), row=2,col = 1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(title="Stock Summary Plot", yaxis_title="Close Price")
    fig.update_layout(width = 1000 , height = 600)
    st.plotly_chart(fig)
          
    if ticker != '-':
	    Summary = GetSummary(ticker)
	    Summary = Summary.set_index('attribute')
	    Summary["value"] = Summary["value"].astype(str)
	    col1.dataframe(Summary, height = 1000)
        
    
    
    @st.cache
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')
    st.download_button(label="Download Summary",data=convert_df_to_csv(Summary),file_name='StockSummary.csv',mime='text/csv',)
    
#==============================================================================
# Tab 2
#==============================================================================

def tab2():
    
    #Dashboard Title
    annotated_text(("Stock Analysis","Chart","#3498DB"))

    
    # Add table to show stock data
    @st.cache
    def GetStockData(tickers, start_date, end_date):
        return pd.concat([si.get_data(tick, start_date, end_date) for tick in tickers])
    
    col1,col2 = st.columns([2,2])
    
    #To select Period
    select_data =  ['1mo', '3mo','6mo','ytd','1y','2y','5y','max']
    default  = select_data.index('1y')
    select_Period =  col1.selectbox("Select Period", select_data,index = default)
    
    #To Select interval
    select_interval = ['1d','1mo']
    interval= col2.selectbox("Select Interval", select_interval)
    
    #To select Graph
    select_graph = st.radio("Select Graph", ["Line","Candle"])
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
 
    #getting the stock data 
    data = yf.download(ticker, period = select_Period,interval = interval)
    
    data['diff'] = data['Close'] - data['Open']
    data.loc[data['diff']>=0, 'color'] = 'green'
    data.loc[data['diff']<0, 'color'] = 'red'
    
    
    
    if ticker != '-':
        stock_price = GetStockData([ticker], start_date, end_date)
        
    #check box to display the data
    show_data = st.checkbox("Show data")  
        
    if show_data:
            st.write('Stock price data')
            st.dataframe(stock_price)
            
    if select_graph == "Line":     
       if ticker != '-':
           data = yf.download(ticker, period = select_Period,interval = interval)
           
           fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.07, subplot_titles=('Stock Trend', 'Volume'), 
               row_width=[0.2, 0.7])
           fig.add_trace(go.Scatter(x=data.index, y=data['Close'],name="Stock Trend",showlegend=True),row= 1,col = 1)
           fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True),row=2,col = 1)
           fig.update(layout_xaxis_rangeslider_visible=False)
           fig.update_layout(title="Stock Summary Line Plot", yaxis_title="Close Price")
           fig.update_layout(width = 1000 , height = 600)
           st.plotly_chart(fig)

    elif select_graph == "Candle":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03, subplot_titles=('Stock Trend', 'Volume'), 
               row_width=[0.2, 0.7])

        #candlestick
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                        low=data["Low"], close=data["Close"], name="Stock Trend"), 
                        row=1, col=1)
        fig.update_layout(title="Stock Summary Candlestick Plot", yaxis_title="Close Price")
        
        #Volume
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index,y=data['Close'].rolling(window=50).mean(),marker_color='orange',name='50 Day MA'))
        #This removes rangeslider 
        fig.update(layout_xaxis_rangeslider_visible=False)
    
        fig.update_layout(
        width=1000,
        height=600,
        autosize=False,
        template="plotly_white")
        st.plotly_chart(fig)



#==============================================================================
# Tab 3 - Statistics
#==============================================================================       
def tab3():
     
#Dashboard Header 
    annotated_text(("Stock","Statistics","#3498DB"))
    
    
# Getting stock data
    def GetStatsEval(ticker):
        return si.get_stats_valuation(ticker)
    def GetStats(ticker):
        return si.get_stats(ticker)
    
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')
    
    
    if ticker != '-':
        statsval = GetStatsEval(ticker)
        statsval = statsval.rename(columns={0:'Valuation Measures',1:'USD'})
        
        #Valuation Measures
        annotated_text(("VALUATION","MEASURES","#3498DB"))
        st.dataframe(statsval,height = 1000)
    #Get Remaining stats
    if ticker != '-':
        stat = GetStats(ticker)
        stat = stat.set_index('Attribute')
        
        #stock Price History
        annotated_text(("STOCK PRICE","HISTORY","#3498DB"))
        Sph = stat.iloc[0:7,]
        st.dataframe(Sph,height = 1000)
        
        #share statistics
        annotated_text(("SHARE","STATISTICS","#3498DB"))
        Shs = stat.iloc[7:19,]
        st.dataframe(Shs,height = 1000)
        
        #Dividend & Splits
        annotated_text(("DIVIDEND","SPLITS","#3498DB"))
        Div = stat.iloc[19:29,]
        st.table(Div)
        
        #Financial Highlights
        annotated_text(("FINANCIAL","HIGHLIGHTS","#3498DB"))
        Finh = stat.iloc[29:31,]
        st.table(Finh)
        
        #Profitability
        annotated_text(("STOCK","PROFITABILITY","#3498DB"))
        Prof = stat.iloc[31:33,]
        st.dataframe(Prof,height = 1000)
        
        #Management Effectiveness
        annotated_text(("Management","Effectiveness","#3498DB"))
        Meff = stat.iloc[33:35,]
        st.dataframe(Meff,height = 1000)
        
        #Income Statement
        IncS = stat.iloc[35:43,]
        annotated_text(("INCOME","STATEMENT","#3498DB"))
        st.dataframe(IncS,height = 1000)
        
        #Balance Sheet
        annotated_text(("BALANCE","SHEET","#3498DB"))
        BalS = stat.iloc[43:49,]
        st.dataframe(BalS,height = 1000)
        
        #Cash Flow
        annotated_text(("CASH","FLOW","#3498DB"))
        Caf = stat.iloc[49:51,]
        st.dataframe(Caf,height = 1000)
        
        
        
# =============================================================================
#         df = stat.style.set_properties(**{'background-color': 'black',
#                            'color': 'lawngreen',
#                            'border-color': 'white'})
# =============================================================================

    #Download Required Data 
    data_to_download = ["Valuation Measures","stock Price History","share statistics","Dividend & Splits","Financial Highlights",
                        "Profitability","Management Effectiveness","Income Statement","Balance Sheet","Cash Flow"]
    to_download = st.selectbox("Choose Data to Download", data_to_download)    
    
    #Conditions to selecr the
    if to_download == 'Valuation Measures':
           st.download_button(label="Download Stats",data=convert_df_to_csv(statsval),file_name='ValuationMeasures.csv',mime='text/csv',)
    elif to_download == 'stock Price History':
         st.download_button(label="Download Stats",data=convert_df_to_csv(Sph),file_name='stockPriceHistory.csv',mime='text/csv',)
    elif to_download == 'share statistics':
            st.download_button(label="Download Stats",data=convert_df_to_csv(Shs),file_name='shareStatistics.csv',mime='text/csv',)
    elif to_download == 'Dividend & Splits':
         st.download_button(label="Download Stats",data=convert_df_to_csv(Div),file_name='DividendAndSplits.csv',mime='text/csv',)
    elif to_download == 'Financial Highlights':
            st.download_button(label="Download Stats",data=convert_df_to_csv(Finh),file_name='FinancialHighlights.csv',mime='text/csv',) 
    elif to_download == 'Profitability':
         st.download_button(label="Download Stats",data=convert_df_to_csv(Prof),file_name='Profitability.csv',mime='text/csv',)
    elif to_download == 'Management Effectiveness':
            st.download_button(label="Download Stats",data=convert_df_to_csv(Meff),file_name='ManagementEffectiveness.csv',mime='text/csv',)
    elif to_download == 'Income Statement':
         st.download_button(label="Download Stats",data=convert_df_to_csv(IncS),file_name='IncomeStatement.csv',mime='text/csv',)
    elif to_download == 'Balance Sheet':
            st.download_button(label="Download Stats",data=convert_df_to_csv(BalS),file_name='BalanceSheet.csv',mime='text/csv',)
    elif to_download == 'Cash Flow':
            st.download_button(label="Download Stats",data=convert_df_to_csv(Caf),file_name='CashFlow.csv',mime='text/csv',)
        
              
    
                
    
    

#==============================================================================
# Tab 4 - Financials 
#==============================================================================       
def tab4():
     
     annotated_text(("Stock", "Financials","#33ADFF"))
     col1,col2 = st.columns([2,2])
     select_data =  ['Income Statement', 'Balance Sheet','Cash Flow']
     fin =  col1.selectbox("Select data", select_data)
     select_term = ['Annual', 'Quarterly']
     term= col2.selectbox("Select Term", select_term)
     
     if ticker != '-':
            if fin == 'Income Statement' and term == 'Quarterly':
                st.table(si.get_income_statement(ticker, yearly=False))
            elif fin == 'Income Statement' and term == 'Annual':
                st.table(si.get_income_statement(ticker))
                
            elif fin == 'Balance Sheet' and term == 'Quarterly':
                st.table(si.get_balance_sheet(ticker, yearly=False))
            elif fin == 'Balance Sheet' and term == 'Annual':
                    st.table(si.get_balance_sheet(ticker))

            elif fin == 'Cash Flow' and term == 'Quarterly':
                    st.table(si.get_cash_flow(ticker, yearly=False))
            elif fin == 'Cash Flow' and term == 'Annual':
                    st.table(si.get_cash_flow(ticker))
                     
#==============================================================================
# Tab 5 -  Analysis
#==============================================================================              
def tab5():
    
    
    annotated_text(("Stock", "Analysis","#33ADFF"))
    
    # Add table to show stock data
    @st.cache
    def GetAns(tickers):
        return ([si.get_analysts_info(tick) for tick in tickers])
    
        
    if ticker != '-':
        analysis = si.get_analysts_info(ticker)
             
        Earnings_Estimate = analysis['Earnings Estimate']
        Revenue_Estimate = analysis['Revenue Estimate']
        Earnings_History = analysis['Earnings History']
        EPS_Trend = analysis['EPS Trend']
        EPS_Revisions = analysis['EPS Revisions']
        Growth_Estimates = analysis['Growth Estimates']
        
        data = [Earnings_Estimate,Revenue_Estimate,
                Earnings_History,EPS_Trend,EPS_Revisions,Growth_Estimates]
        for i in data:
            st.table(i)
             
            
             
#==============================================================================
# Tab 6 -  Monte Carlo Simulation 
#==============================================================================       
def tab6():
    
    #Title for Montecarlo
    annotated_text(("Monte Carlo" ,"Simulation","#33ADFF"))
    
    #Dropdown to select number of simulations & time horizon
    col1,col2 = st.columns([2,2])
    select_sims =  [200, 500,1000]
    sim =  col1.selectbox("Select simulations", select_sims)
    select_horizon = [30, 60,90]
    horizon= col2.selectbox("Select Horizon", select_horizon)
    
    
    
    # Getting the Required stock price from Yahoo finance
    stock_price = web.DataReader(ticker, 'yahoo',start_date, end_date)
   
    #close price, daily return & daily volatility 
    close_price = stock_price['Close']
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)

    
    #Monte Carlo simulation
    np.random.seed(9)
    simulations = sim
    time_horizone = horizon

    # Run the simulation
    simulation_df = pd.DataFrame()

    for i in range(simulations):
    
    # The list to store the next stock price
       next_price = []
    
       #Next stock price
       last_price = close_price[-1]
    
       for j in range(time_horizone):
           #Future return around the mean (0) and daily_volatility
           future_return = np.random.normal(0, daily_volatility)

           #Random future price
           future_price = last_price * (1 + future_return)

           # Save the price and go next
           next_price.append(future_price)
           last_price = future_price
    
       # Store the result of the simulation
       simulation_df[i] = next_price
    
    # Plotting the simulation stock price in the future
    #mfig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)

    plt.plot(simulation_df)
    plt.title('Monte Carlo simulation')
    plt.xlabel('Day')
    plt.ylabel('Price')

    plt.axhline(y=close_price[-1], color='red')
    plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    # Ending price 
    ending_price = simulation_df.iloc[-1:, :].values[0, ]
    
    # Stock Price at 95% confidence interval
    future_price_95ci = np.percentile(ending_price, 5)
    
    # Finding out Value at Risk(VAR)
  
    VaR = close_price[-1] - future_price_95ci
    st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')  
    
        
#==============================================================================
# Main body
#==============================================================================

def run():
    
    st.markdown("""<p style="text-align: center;"><span style="color: rgb(44, 130, 201); font-size: 40px;"><strong>Yahoo Finance Stocks</strong></span></p>""", unsafe_allow_html=True)    
    annotated_text(("Data source:","Yahoo Finance","#33FFC0"))
     
    
    #Ticker selection on the sidebar
    # Getting the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    default = ticker_list.index('NFLX')
    
    
    #selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list,index = default)
    
  
    #select Start & end dates
    global start_date, end_date
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    
    
    #Refresh the form
    with st.sidebar:
         with st.form(key = "Refresh Form"):
              st.form_submit_button(label = "Update")
                      
    #radio box to select the tabs 
    select_tab = st.sidebar.selectbox("Select tab", ['Summary','Chart','Statistics','Financials','Analysis','Monte Carlo Simulation'])
    
    #Display the selected tab
    if select_tab == 'Summary':
        tab1()
        
    elif select_tab == 'Chart':
        tab2()
        
    elif select_tab == 'Statistics':
        tab3()
        
    elif select_tab == 'Financials':
        tab4()
        
    elif select_tab == 'Analysis':
        tab5()
        
    elif select_tab == 'Monte Carlo Simulation':
        tab6()
    
    
if __name__ == "__main__":
    run()
    
#ThE eNd