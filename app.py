import os
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from functools import reduce
from dateutil import parser
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
# Title of the Streamlit app
st.title("CSV Data Viewer")

# GitHub raw file path to auto-load
default_file_path = (
    "https://raw.githubusercontent.com/iblend171/Trade_Lots_POC/main/OPEN_ind_tsx_350list.csv"
)

# Function to load the CSV from the URL
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Load the CSV and display it


def elim_duplicates(fn):
    new_list = []
    for i in range(len(fn)):
        if (fn[i][4] != fn[i-1][4]):
            new_list.append(fn[i])
    return new_list

def plusbox(dataframe):
# Iterate in reverse to find latest plus box
    boxes = []
    m = dataframe.shape[0]
    for t in reversed(range(10,m)):
        
    # Find max high for 11 periods using iterator t and compare with latest period high
        date = dataframe['Date'].iloc[t].date()
        max_array = dataframe['High'].iloc[t-10:t+5].max()
        # Find local high event trigger
        if (dataframe['High'].iloc[t] == max_array) and (t < (m-4)):
            period_start = dataframe['High'].iloc[t]
#             print(period_start)
            trend = dataframe['Low'].iloc[t+4]
            # Start iterating after 5 period box fix to find if trend is lower than box(5 period) low
            boxes.append([1, 1, date, t, period_start, dataframe['Low'].iloc[t:t+5].min(), 5, dataframe['Date'].iloc[t + 4]])
#             print(1, 1, date, t, period_start, dataframe['Low'].iloc[t:t+5].min(), 5, dataframe['Date'].iloc[t + 4])   
            for k in range(5,m-t):
#                 print(t)
            # box closes if trend reverses...
                if ((dataframe['Low'].iloc[t+k]) > trend):
                # compare trend with 5 period box high
                    if (dataframe['Low'].iloc[t:t+5].min() >= trend):
                        boxes.append([0, 1, date, t, period_start, trend, k, dataframe['Date'].iloc[t + k]])
#                         print(0, 1, date, t, period_start, trend, k, dataframe['Date'].iloc[t + k])
                        break
                    elif (dataframe['Low'].iloc[t:t+5].min() < trend):
                        boxes.append([0, 1, date, t, period_start, dataframe['Low'].iloc[t:t+5].min(), k, dataframe['Date'].iloc[t + k]])
#                         print(0, 1, date, t, period_start, dataframe['Low'].iloc[t:t+5].min(), k, dataframe['Date'].iloc[t + k])
                        break
                elif (dataframe['Low'].iloc[t+k] <= trend):
                # update trend for new lows
                    trend = dataframe['Low'].iloc[t+k]

    return (boxes)

def minusbox(dataframe):
# Iterate in reverse to find latest plus box
    m = dataframe.shape[0]
    boxes = []
    for t in reversed(range(10,m)):
    # Find min low for 11 periods using iterator t and compare with latest period low
        date = dataframe['Date'].iloc[t].date()
        min_array = dataframe['Low'].iloc[t-10:t+5].min()
        if (dataframe['Low'].iloc[t] == min_array) and (t < (m-4)):
#             < dataframe[(dataframe['Instrument'] == instrument)]['Low'].iloc[t:t+5].min()
        # if conditions are satisfied fix box start and trend
            period_start = dataframe['Low'].iloc[t]
            trend = dataframe['High'].iloc[t+4]
            # start iterating after 5 period box fix to find if trend is higher than box(5 period) high
            boxes.append([0, 0, date, t, dataframe['High'].iloc[t:t+5].max(), period_start,  5, dataframe['Date'].iloc[t + 4]])
#             print(0, 0, date, t, dataframe['High'].iloc[t:t+5].max(), period_start,  5, dataframe['Date'].iloc[t + 4])
            for k in range(5,m-t):
            # box closes if trend reverses...
                if (dataframe['High'].iloc[t+k] < trend):
                # compare trend with 5 period box high
                    if (dataframe['High'].iloc[t:t+5].max() <= trend):
                        boxes.append([1, 0, date, t, trend, period_start, k, dataframe['Date'].iloc[t + k]])
#                         print(1, 0, date, t, trend, period_start, k, dataframe['Date'].iloc[t + k])
                        break
                    elif (dataframe['High'].iloc[t:t+5].max() > trend):
                        boxes.append([1, 0, date, t, dataframe['High'].iloc[t:t+5].max(), period_start, k, dataframe['Date'].iloc[t + k]])
#                         print(1, 0, date, t, dataframe['High'].iloc[t:t+5].max(), period_start, k, dataframe['Date'].iloc[t + k])
                        break
                elif (dataframe['High'].iloc[t+k] >= trend):
                # update trend for new lows
                    trend = dataframe['High'].iloc[t+k]

    return (boxes)
#-----^-------^------^--------^------^-------CHANGED FOR SHORTING----^-------^------^--------^------^-------

wait = 11
def box_break(box_df, df):
#   Waiting for the box to break
    list_main = []
    for i in range(0, box_df.shape[0]):
        row_list = box_df.iloc[i].values.flatten().tolist()
#       # Because box length is int value that includes start box index i.e. starts from 1 not zero
        box_close = box_df['Record'].iloc[i] + box_df['Length'].iloc[i] 
        # print("Hey")
        if box_df['Trade'].iloc[i] == 1:
        # Find Max values reached in the next two weeks i.e. 10 periods
            max_value = df['High'][box_close+1:box_close + wait].max()
            min_value = df['Low'][box_close+1:box_close + wait].min()
            # Minus 1 percent profit
            minus_one = (box_df['Resistance'].iloc[i] - (box_df['Resistance'].iloc[i] * 0.01))
            profit_point_five = (minus_one + (0.005*minus_one))
            # Stop Loss @ 2 percent
            stop_loss = minus_one - (minus_one*0.02)
            if (max_value >= profit_point_five) & (df[df['High'] == max_value]['Date'] > box_df['Close_Date'].iloc[i]).any() :
                profit = (max_value-minus_one)
                profit_pct = ((max_value-minus_one)/minus_one)*100
                row_list = row_list + [max_value, min_value, df.iloc[box_close+1:box_close + wait][df['High'] >= max_value]['Date'].to_list()[0].date(),
                                       df.iloc[box_close+1:box_close + wait][df['Low'] <= min_value]['Date'].to_list()[0].date(),
                                       df.iloc[box_close+1:box_close + wait][df['High'] >= max_value].index[0], round(profit,2), round(profit_pct,2), 1]
                list_main.append(row_list)
            elif (max_value < profit_point_five) & (max_value >= minus_one) & (df[df['High'] == max_value]['Date'] > box_df['Close_Date'].iloc[i]).any() :
                profit = (max_value-minus_one)
                profit_pct = ((max_value-minus_one)/minus_one)*100
                row_list = row_list + [max_value, min_value, df.iloc[box_close+1:box_close + wait][df['High'] >= max_value]['Date'].to_list()[0].date(), 
                                       df.iloc[box_close+1:box_close + wait][df['Low'] <= min_value]['Date'].to_list()[0].date(),
                                       df.iloc[box_close+1:box_close + wait][df['High'] >= max_value].index[0], round(profit,2), round(profit_pct,2), 0]
                list_main.append(row_list)
        else:
            # Find Min values reached in the next two weeks i.e. 10 periods
            max_value = df['High'][box_close+1:box_close + wait].max()
            min_value = df['Low'][box_close+1:box_close + wait].min()
            # Plus 1 percent profit
            plus_one = box_df['Support'].iloc[i] + (box_df['Support'].iloc[i] * 0.01)
            profit_point_five = plus_one - (0.005*plus_one)
            # Stop Loss @ 2 percent
            stop_loss = plus_one + (plus_one*0.02)
            if (min_value <= profit_point_five) & (df[df['Low'] == min_value]['Date'] > box_df['Close_Date'].iloc[i]).any() :
                profit = (plus_one - min_value)
                profit_pct = ((plus_one - min_value)/plus_one)*100
                row_list = row_list + [min_value, max_value, df.iloc[box_close:box_close + wait][df['Low'] <= min_value]['Date'].to_list()[0].date(),
                                       df.iloc[box_close+1:box_close + wait][df['High'] >= max_value]['Date'].to_list()[0].date(), 
                                       df.iloc[box_close:box_close + wait][df['Low'] <= min_value].index[0], round(profit,2), round(profit_pct,2), 1]
                list_main.append(row_list)
            elif (min_value > profit_point_five) & (min_value <= plus_one) & (df[df['Low'] == min_value]['Date'] > box_df['Close_Date'].iloc[i]).any() :
                profit = (plus_one - min_value)
                profit_pct = ((plus_one - min_value)/plus_one)*100
                row_list = row_list + [min_value, max_value, df.iloc[box_close+1:box_close + wait][df['Low'] <= min_value]['Date'].to_list()[0].date(), 
                                       df.iloc[box_close+1:box_close + wait][df['High'] >= max_value]['Date'].to_list()[0].date(), 
                                       df.iloc[box_close:box_close + wait][df['Low'] <= min_value].index[0], round(profit,2), round(profit_pct,2), 0]
                list_main.append(row_list)
    return list_main

def box_break_1pct(pl_df, df):
    list_main = []
    for i in range(0,pl_df.shape[0]):
        
        # Calculation starts next day i.e. 1 period after close date
        box_close = pl_df['Record'].iloc[i] + pl_df['Length'].iloc[i]
        max_value = df['High'][box_close+1:box_close + wait].max()
        min_value = df['Low'][box_close+1:box_close + wait].min()

        if (pl_df.iloc[i]['Trade'] == 1):
            try:
                one_pct_date = df.iloc[box_close+1:box_close + wait][df['High'] >= pl_df.iloc[i]['Resistance']]['Date'].to_list()[0].date()
                list_main.append(one_pct_date)
#                 print(i, "Long YES!", pl_df.iloc[i]['Resistance'], box_close, one_pct_date)
            except:
#                 print(box_close,i, "Failed",1)
                list_main.append(pl_df.iloc[i]['Max_Profit_Date'])
#                 print(i, "Long NO!", pl_df.iloc[i]['Resistance'], box_close, pl_df.iloc[i]['Max_Profit_Date'])
        elif (pl_df.iloc[i]['Trade'] == 0):
            try:
                one_pct_date = df.iloc[box_close+1:box_close + wait][df['Low'] <= pl_df.iloc[i]['Support']]['Date'].to_list()[0].date()
#                 print(i, "Short YES!", pl_df.iloc[i]['Support'], box_close, one_pct_date)
                list_main.append(one_pct_date)
#                 print(box_close, i,one_pct_date,0)
            except:
#                 print(i, "Short NO!", pl_df.iloc[i]['Support'], box_close, pl_df.iloc[i]['Max_Profit_Date'])
#                 print(box_close, i, "Failed",0)
                list_main.append(pl_df.iloc[i]['Max_Profit_Date'])
    return list_main

def box_fresh(box_df, df):
    list_main = []
    for i in range(0,box_df.shape[0]):
#         print(i)
#     ALL Close dates from boxes_df have to be less than last recorded date from raw data df
        if box_df['Close_Date'].iloc[i] < df['Date'].iloc[-1]:
            start_scan = df[(df['Date'] > box_df['Close_Date'].iloc[i])].index[0]
#             print(i,box_df['Close_Date'].iloc[i], start_scan)
            # Check Freshness of Trigger!
            if boxes_df['Trade'].iloc[i] == 1:
#                 print(i, box_df['Close_Date'].iloc[i], (df.iloc[start_scan:]['High'] > box_df['Entry'].iloc[i]).sum())
                v = (df.iloc[start_scan:]['High'] > box_df['Entry'].iloc[i]).sum()
                list_main.append(v)
            elif boxes_df['Trade'].iloc[i] == 0:
#                 print(i, box_df['Close_Date'].iloc[i], (df.iloc[start_scan:]['Low'] < box_df['Entry'].iloc[i]).sum())
                v = (df.iloc[start_scan:]['Low'] < box_df['Entry'].iloc[i]).sum()
                list_main.append(v)
        else:
            list_main.append(0)
    return list_main

def box_open(box_df, df):
#   Checking if the box is open
    list_main = []
    for i in range(0, box_df.shape[0]):
        row_list = box_df.iloc[i].values.flatten().tolist()
        list_main.append(row_list)
        
    return list_main

start=date(2021,10,1)
end=date(2025,1,1)

# ------------------------------------ALL STOCKS LISTED FOR INDEX------------------------------------

# create an Empty DataFrame object
df_ful = pd.DataFrame(columns = ['Trade', 'Polarity', 'Date', 'Record', 'Resistance', 'Support', 'Length', 'Close_Date', 
                                 'Amplitude', 'Entry', 'Stop_Loss', 'Symbol', 'currency', 'exchange', 'company_name', 'company_id'])
df_lul = pd.DataFrame(columns = ['Trade', 'Polarity', 'Date','Record','Resistance','Support', 'Length', 'Close_Date', 'Amplitude','Entry', 'Stop_Loss',\
                                     'CMP','CMP_Ready', 'Max_Profit','Max_Loss','Max_Profit_Date','Max_Loss_Date','Trade_Record',  'Profit', 'Profit_%',  'P/L',
                                  'Symbol', 'currency', 'exchange', 'company_name', 'company_id'])

tsx_350 = pd.read_csv(r"TSX_100.csv")

# tsx_350 = ddf['Symbol'].to_list()
cut_off = 1
fail_list = []
for t in tsx_350['Symbol'].to_list()[:10]:
    try:
        print(t)
        df = yf.download(t +'.TO', start=start, end=end, progress=False)
        # Remove the multi-index from columns by dropping the 'Ticker' level
        df.columns = df.columns.droplevel(1)
        df['Date'] =  pd.to_datetime(df.index)
        df.index = pd.RangeIndex(len(df.index))
        df = df.infer_objects()


    #     df.drop_duplicates(subset=['Volume'], keep='first', inplace =True)
    #     print(df.head())
        train = []

        train.append(plusbox(df))
        train.append(minusbox(df))

        # Reshape list to 1D
        train = reduce(lambda x,y :x+y ,train)
        entry = 0.01
        stop_loss = 0.02
        boxes_df = pd.DataFrame(train, columns=['Trade','Polarity', 'Date','Record','Resistance','Support','Length', 'Close_Date']).sort_values(by = 'Date', ascending = False)
        boxes_df['Amplitude'] = round(((abs(boxes_df['Resistance'] - boxes_df['Support'])/boxes_df['Support'])*100),2)

        boxes_df['Entry'] = np.where(boxes_df['Trade'] == 1, round(boxes_df['Resistance'] - (entry*boxes_df['Resistance']),4)
                                        , round(boxes_df['Support'] + (entry*boxes_df['Support']),4))

        boxes_df['Stop_Loss'] = np.where(boxes_df['Trade'] == 1, round(abs(boxes_df['Entry'] - stop_loss*(abs(boxes_df['Entry']))),4)
                                        , round(abs(boxes_df['Entry'] + stop_loss*(abs(boxes_df['Entry']))),4))

        # boxes_df['Yearly_High'] = df[df['Date'].dt.date >= (datetime.now().date() - timedelta(365))]['High'].max()
        # boxes_df['Yearly_Low'] = df[df['Date'].dt.date >= (datetime.now().date() - timedelta(365))]['Low'].min()


        # boxes_df['Yearly_High_Date'] = df.loc[df['High'] == (df[df['Date'].dt.date >= 
        #                                                         (datetime.now().date() - timedelta(365))]['High'].max())]['Date'][-1:].item().date()
        # boxes_df['Yearly_Low_Date'] = df.loc[df['Low'] == (df[df['Date'].dt.date >= 
        #                                                     (datetime.now().date() - timedelta(365))]['Low'].min())]['Date'][-1:].item().date()
        # boxes_df['CMP'] = df['Close'].iloc[-1]
        # boxes_df['Volume'] = df['Volume'].iloc[-1]

        # print(boxes_df.head(1))
        #Ready Percentage 
        rp = 0.03  
        # boxes_df['CMP_Ready'] = np.where((boxes_df['CMP'] >= (boxes_df['Entry'] - rp*(boxes_df['Entry'])))
    #                                      & ((df['High'].iloc[-5:] < boxes_df['Entry'].iloc[0]).all())
                                            # & (boxes_df['CMP'] < (boxes_df['Entry'])), 1, 0)

        # boxes_df['Fresh_Trigger'] = box_fresh(boxes_df, df)
        lulu = box_break(boxes_df, df)
        fulu = box_open(boxes_df, df)
        # print(lulu[0])

        # Compute Closed Boxes
        pl_df = pd.DataFrame(lulu, columns = ['Trade', 'Polarity', 'Date','Record','Resistance','Support', 'Length', 'Close_Date',
                                            'Amplitude','Entry', 'Stop_Loss', 'Max_Profit', 'Max_Loss','Max_Profit_Date',
                                            'Max_Loss_Date', 'Trade_Record', 'Profit', 'Profit_%', 'P/L'])
        # print(pl_df)
        pl_df['Date'] =  pd.to_datetime(pl_df['Date'])
        pl_df['Symbol'] = t +'.TO'
        pl_df['currency'] = "CAD"
        pl_df['exchange'] = 2
        pl_df['company_name'] = tsx_350[tsx_350['Symbol'] == t]['Company Name'].item()
        pl_df['company_id'] = tsx_350.loc[tsx_350['Symbol'] == t].index[0] + 1
        # Total Events, Total Profit, Total Loss
        pl_df["Total_Events"] = np.nan   
        pl_df['Total_Profit_Events'] = np.nan
        pl_df['Total_Loss_Events'] = np.nan
        for i in range(0, pl_df.shape[0]):   
        #     print(pl_df['P/L'].iloc[i:].sum(), pl_df.index[i:].shape[0], pl_df.iloc[:-i][pl_df['P/L'] == 0].count())
            pl_df['Total_Events'].iloc[i] = pl_df.index[i:].shape[0]
            pl_df['Total_Profit_Events'].iloc[i] = pl_df['P/L'].iloc[i:].sum()
            pl_df['Total_Loss_Events'].iloc[i] = pl_df.iloc[i:][pl_df['P/L']==0]['P/L'].count()

        pl_df["Total_Events"] = pl_df['Total_Events'].astype(int)
        pl_df['Total_Profit_Events'] = pl_df['Total_Profit_Events'].astype(int)
        pl_df['Total_Loss_Events'] = pl_df['Total_Loss_Events'].astype(int)
        pl_df['Profit_Ratio'] = pl_df['Total_Profit_Events']/pl_df['Total_Events']

        # Profit_Profile
        pl_df['Min_Profit%'] = pl_df['Profit_%'].min()
        pl_df['Max_Profit%'] = pl_df['Profit_%'].max()

        pl_df['Profit_0.5%'] = pl_df['Profit_%'].groupby(pd.cut(pl_df['Profit_%'], [0.5,1])).count().iloc[0]
        pl_df['Profit_1.0%'] = pl_df['Profit_%'].groupby(pd.cut(pl_df['Profit_%'], [1,1.5])).count().iloc[0]
        pl_df['Profit_1.5%'] = pl_df['Profit_%'].groupby(pd.cut(pl_df['Profit_%'], [1.5,2])).count().iloc[0]
        pl_df['Profit_2.0%'] = pl_df['Profit_%'].groupby(pd.cut(pl_df['Profit_%'], [2,5])).count().iloc[0]
        pl_df['Profit_5.0%'] = pl_df['Profit_%'].groupby(pd.cut(pl_df['Profit_%'], [5,float('inf')])).count().iloc[0]

        # Loss_Profile
        v = ((datetime.now() - pl_df[pl_df['P/L'] == 0]['Date']).dt.days/30).astype(int)

        pl_df['Loss_Profile_3M'] = np.where((v <= 3), 1, 0).sum()
        pl_df['Loss_Profile_6M'] = np.where(((v > 3) & (v <= 6)), 1, 0).sum()
        pl_df['Loss_Profile_12M'] = np.where(((v > 6) & (v <= 12)), 1, 0).sum()
        pl_df['Loss_Profile>12M'] = np.where((v > 12) , 1, 0).sum()

        # One_%_Profit Date >> REAL BOOK PROFIT DATE<<
        pl_df['One_%_Profit_Date'] = box_break_1pct(pl_df, df)

        # Stop Loss Analysis

        pl_df['Stop_Loss_2%'] = pl_df[(pl_df['Max_Loss'] <= (pl_df['Entry']- 0.02*(pl_df['Entry']))) 
                                        & (pl_df['Max_Loss_Date'] < pl_df['One_%_Profit_Date'])].shape[0]
        pl_df['Stop_Loss_5%'] = pl_df[(pl_df['Max_Loss'] <= (pl_df['Entry']- 0.05*(pl_df['Entry']))) 
                                        & (pl_df['Max_Loss_Date'] < pl_df['One_%_Profit_Date'])].shape[0]
        pl_df['Stop_Loss_7%'] = pl_df[(pl_df['Max_Loss'] <= (pl_df['Entry']- 0.07*(pl_df['Entry']))) 
                                        & (pl_df['Max_Loss_Date'] < pl_df['One_%_Profit_Date'])].shape[0]
        pl_df['Stop_Loss_10%'] = pl_df[(pl_df['Max_Loss'] <= (pl_df['Entry']- 0.1*(pl_df['Entry']))) 
                                        & (pl_df['Max_Loss_Date'] < pl_df['One_%_Profit_Date'])].shape[0]


        df_lul = pd.concat([df_lul, pl_df], sort=False)

        # Compute Open Boxes

        fulu_df = pd.DataFrame(fulu, columns = ['Trade', 'Polarity', 'Date','Record','Resistance','Support', 'Length', 'Close_Date',
                                            'Amplitude','Entry', 'Stop_Loss'])

        fulu_df['Date'] =  pd.to_datetime(fulu_df['Date'])
        fulu_df['Symbol'] = t +'.TO'
        fulu_df['currency'] = "CAD"
        fulu_df['exchange'] = 2
        fulu_df['company_name'] = tsx_350[tsx_350['Symbol'] == t]['Company Name'].item()
        fulu_df['company_id'] = tsx_350.loc[tsx_350['Symbol'] == t].index[0] + 1

        df_ful = pd.concat([df_ful, fulu_df], sort=False)
        # print(pl_df)
        
    except:
        fail_list.append(t)
#         print("FAIL: ", t +'.TO')
        continue

# print(df_lul.head())
# print(df_ful.head())


# Step 1: Select the first entry for each 'Symbol' in df_lul
df_lul_first = df_lul.groupby('Symbol').first().reset_index()

# Step 2: Select only the relevant columns
df_lul_first = df_lul_first[['Symbol', 'Total_Events', 'Total_Profit_Events', 'Total_Loss_Events','Profit_Ratio']]

# Step 3: Merge with df_ful on 'Symbol'
df_merged = df_ful.merge(df_lul_first, on='Symbol', how='right')

# Step 4: Select the first entry for each 'Symbol' in df_merged
df_select = df_merged.groupby('Symbol').first().reset_index()

# Define the bins and labels for the ranges
bins = [0.85, 0.9, 0.95, 1.0]
labels = ['0.85-0.9', '0.9-0.95', '0.95-1.0']

# Create a new column that categorizes 'Profit_Ratio' based on the bins
df_select['Profit_Ratio_Range'] = pd.cut(df_select['Profit_Ratio'], bins=bins, labels=labels, include_lowest=True)

# Slice the DataFrame based on the new 'Profit_Ratio_Range' column
df_95_100 = df_select[df_select['Profit_Ratio_Range'] == '0.95-1.0']
df_90_95 = df_select[df_select['Profit_Ratio_Range'] == '0.9-0.95']
df_85_90 = df_select[df_select['Profit_Ratio_Range'] == '0.85-0.9']

if df is not None:
    st.subheader("Tabular Data:")
    st.dataframe(df_95_100)
    st.dataframe(df_90_95)
    st.dataframe(df_85_90)

    st.subheader("Summary Statistics:")
    st.write(df_select)

# print(df.head())


os.system("streamlit run app.py")
