from pytrends.request import TrendReq
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import json
import time
import pyinputplus as pyip

# import english_stock_ticker_names.csv, set as df 'names'
names = pd.read_csv('english_stock_ticker_names.csv', index_col=0)
names = names.set_index('numerai_ticker')

# open dictionary with 2004 trends info
with open('trends_2004.json') as f:
    trends_2004_dict = json.load(f)    
names['trends_2004'] = pd.Series(trends_2004_dict)
names

# set error_raised to True for CLF
names.loc['CLF', 'error_raised'] = True

def make_keywords(ticker):
    #keyword_list = [f'{ticker_name_map[ticker]} stock', f'{ticker} stock']
    ticker_cleaned, name = names.loc[ticker].tolist()
    keyword_list = [f'{ticker_cleaned} stock', f'{name} stock']
    return keyword_list
    
def get_overlapping_trends(keyword, first_date=datetime(year=2004, month=1, day=4)):
    
    if first_date.weekday() != 6:
        raise ValueError('first_date is not a Sunday')
    
    pytrend = TrendReq()
    trends_list = []
    start_dates = []
    
    # keep track of whether first 135 weeks have been successfully downloaded
    first_135 = False
    
    while not first_135:
        # get first 135 weeks
        first_plus_135 = first_date + timedelta(weeks=134)
        timeframe = f'{first_date.strftime("%Y-%m-%d")} {first_plus_135.strftime("%Y-%m-%d")}'
        pytrend.build_payload([keyword], timeframe=timeframe)

        print(f'Trying first 135 weeks of trends for \'{keyword}\' from {first_date} to {first_plus_135}')
        trends = pytrend.interest_over_time()

        # if no trends were found, df will be empty. 
        if trends.empty:
            print('None found')
            # shift first_date on 135 weeks for next try
            first_date = first_date + timedelta(weeks=135)
            
            # if no first 135 weeks can be found, return a pair of empty dataframes
            if first_date > datetime.now():
                return [trends, trends]
        else:
            print('Downloaded successfully')
            trends.drop(columns=['isPartial'], inplace=True)
            trends_list.append(trends)
            first_135 = True
    
    next_date = first_date
    while next_date < datetime.now():
        start_date = next_date

        # check this period has not already been downloaded. If it has, break
        if start_date not in start_dates:
            start_dates.append(start_date)
            
            # 270 weeks seems to be the max that works on a weekly basis
            start_date_plus = start_date + timedelta(weeks=269)
            timeframe = f'{start_date.strftime("%Y-%m-%d")} {start_date_plus.strftime("%Y-%m-%d")}'
            pytrend = TrendReq()
            pytrend.build_payload([keyword], timeframe=timeframe)

            print(f'Downloading trends for \'{keyword}\' from {start_date} to {start_date_plus}')
            trends = pytrend.interest_over_time()
            trends.drop(columns=['isPartial'], inplace=True)
            trends_list.append(trends)

            next_date = start_date + timedelta(weeks=135)
        else:
            break
            
    return trends_list
    
def scale(to_scale, scale_by):
    '''
    Scale one series by another
    '''
    factor = scale_by.max() - scale_by.min()
    scaled = factor * (to_scale - to_scale.min()) / (to_scale.max() - to_scale.min())
    scaled += scale_by.min()
    return scaled
    
def index_first(series1, series2):
    # returns index of series1\series2
    idx = [i for i in series1.index if i not in series2.index]
    return idx
    
def overlap(series1, series2):
    # returns index of series1 intersect series2
    idx = [i for i in series1.index if i in series2.index]
    return idx
    
def ext_scale(to_scale, scale_by):
    '''
    Scale one series by another using overlaps
    '''
    
    # find intersections
    inter_ts = to_scale.loc[overlap(to_scale, scale_by)]
    inter_sb = scale_by.loc[overlap(to_scale, scale_by)]
    
    factor = inter_sb.max() - inter_sb.min()
    
    scaled = factor * (to_scale - inter_ts.min()) / (inter_ts.max() - inter_ts.min())
    scaled += inter_sb.min()
    return scaled
    
def rescale_overlaps(output_list):
    # takes a list of overlapping outputs from Google Trends
    # rescales ith output to i-1th using overlaps
    
    ext_scales = []
    
    # first extended scale between output_list[0] and output_list[1]
    ext_scale_1 = ext_scale(output_list[1], output_list[0])
    ext_scales.append(ext_scale_1)
    
    # len - 2 further extended scales between output_list[i] and previous extended scale
    for i in range(2, len(output_list)):
        ext_scale_i = ext_scale(output_list[i], ext_scales[i-2])
        ext_scales.append(ext_scale_i)
        
    return ext_scales
    
def universal_scale(ext_scales):
    # takes the overlapping extended scales and removes rows from ith where it overlaps with i+1th
    # returns one universal scale for the whole time period
    output = []
    
    for i in range(len(ext_scales)-1):
        ext_scale_i = ext_scales[i]
        ext_scale_ip1 = ext_scales[i+1]
        ext_scale_i = ext_scale_i.loc[index_first(ext_scale_i, ext_scale_ip1)]
        output.append(ext_scale_i)
        
    output.append(ext_scales[-1])
    
    # concat elements of output
    output = pd.concat(output)
    
    # scale to 0-100
    output = scale(output, pd.Series([0, 100]))
    return output

def get_trend_history(keyword_list, first_date=datetime(year=2004, month=1, day=4)):
    uni_scales = []
    for keyword in keyword_list:
        # downloads all trends and returns universal scale
        trends = get_overlapping_trends(keyword, first_date)
        ext_scales = rescale_overlaps(trends)
        uni_scale = universal_scale(ext_scales)
        uni_scales.append(uni_scale)
    return pd.concat(uni_scales, axis=1)
    
def ticker_trend(ticker):
    #keyword_list = make_keywords(ticker)
    keyword = names.loc[ticker]['name']
    trend_history = get_trend_history([keyword])
    trend_history['ticker'] = ticker
    
    #mapper = {keyword_list[0]: 'ticker stock', keyword_list[1]: 'name stock'}
    mapper = {keyword: 'name'}
    trend_history = trend_history.rename(columns=mapper)
    return trend_history
    
# get tickers to download trends for (US & trends from 2004 & no error)
print(f'Loading tickers...', end='', flush=True)
names_us_2004 = names[(names['market_country'] == 'us') & (names['trends_2004'] == True) & (names['error_raised'] != True)]
print(f'{len(names_us_2004)} loaded.')

def make_big_ticker_trends(names, trends):
    """
    Get trends for names that are not already in trends
    """
    i = 1
    to_concat = []
    tickers = [ticker for ticker in names.index if ticker not in trends['ticker'].unique()]
    tickers_len = len(tickers)
    for ticker in tickers:
        print(f'{i}/{tickers_len}')
        trend_history = ticker_trend(ticker)
        to_concat.append(trend_history)
        i += 1
        # export CSV every 1
        # if i % 5 == 0:
        print('Exporting...')
        pd.concat([trends] + to_concat).to_csv('trends.csv')
        if i % 5 == 0:
            print('Sleeping for 60 seconds...', end='', flush=True)
            time.sleep(60)
            print('Woken!')
    return pd.concat([trends] + to_concat)
    
print(f'Importing trends.csv...', end='', flush=True)
trends = pd.read_csv('trends.csv', parse_dates = ['date'], index_col='date')
print(f'{len(trends)} rows already obtained across {len(trends["ticker"].unique())} tickers. {len(names_us_2004)-len(trends["ticker"].unique())} remaining.')
#cont = pyip.inputChoice(['y', 'n'], prompt='Continue? y/n...', default='y', timeout=5)
#if cont == 'y' or cont == 'Y':
trends = make_big_ticker_trends(names_us_2004, trends)
#else:
#    print(f'Stopping.')