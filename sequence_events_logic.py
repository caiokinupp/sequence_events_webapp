import yfinance as yf
import pandas as pd

def moving_average(values_series, periods = 20, ma_type = 'sma'):
    """
    Method to calculate SMA or EMA of a series.
    The Series must be ordered from oldest to newest.
    
    Keyword arguments:
    values_series (pandas.Series) -- values to calculate the MA
    periods (integer) -- periods of the calculation (default 20)
    ma_type -- sma or ema (default 'sma')
    """
    
    if ma_type == 'sma':
        return values_series.rolling(periods).mean()
    elif ma_type == 'ema':
        return values_series.ewm(span = periods, min_periods = periods).mean()

    return None


def dist_price_ma(price_series, ma_series):
    """
    Method to calculate the difference (distance) from the price to an MA.
    
    Keyword arguments:
    price_series (pandas.Series) -- price series of values
    ma_series (pandas.Series) -- ma series of values
    """
    
    return price_series - ma_series


def moving_std(values_series, periods = 20):
    """
    Method to calculate the standard deviation of a window of values in a series.
    
    Keyword arguments:
    values_series (pandas.Series) -- series of values to calculate the standard deviation
    periods (integer) -- periods of the calculation (default 20)
    """
    
    return values_series.rolling(periods).std()


def z_score(ma_series, price_series, moving_std_periods):
    """
    Method to calculate the Z-Score.
    Is the distance from the price to a MA in standard deviations.
    (The std is a moving standard deviation).
    
    Keyword arguments:
    ma_series (pandas.Series) -- series of moving average
    price_series (pandas.Series) -- series of values to calculate the standard deviation
    moving_std_periods (integer) -- periods for the calculation of moving std (default 20)
    """
    
    # Calculating distance between price and the MA
    distance_price_ma = dist_price_ma(price_series, ma_series)
    # Calculating the moving standard deviation
    stds_price = moving_std(price_series, moving_std_periods)
    
    return distance_price_ma/stds_price


def set_high_low(values_series):
    """
    Method to define whether a day is bullish or bearish.
    2 -> Lateral
    1 -> Bullish
    0 -> Bearish
    
    Keyword arguments:
    values_series (pandas.Series) -- series of values to define whether a day is bullish or bearish
    """
    
    lst_days_direction = []
    
    for i in values_series.index:
        if i == 0:
            lst_days_direction.append(2)
        else:
            if values_series.iloc[i] > values_series.iloc[i-1]:
                lst_days_direction.append(1)
            elif values_series.iloc[i] < values_series.iloc[i-1]:
                lst_days_direction.append(0)
            else:
                lst_days_direction.append(2)
    
    return pd.Series(lst_days_direction)


def days_in_sequence(direction_series):
    """
    Method to count sequence of days with the same direction.
    
    Keyword arguments:
    direction_series (pandas.Series) -- series of values that define the direction of the day
    """
    
    lst_days_sequence = []
    
    for i in direction_series.index:
        if i == 0:
            sequence_counter = 1
            lst_days_sequence.append(sequence_counter)
        else:
            if direction_series.iloc[i] == direction_series.iloc[i-1]:
                sequence_counter = sequence_counter + 1
                lst_days_sequence.append(sequence_counter)
            else:
                sequence_counter = 1
                lst_days_sequence.append(sequence_counter)
    
    return pd.Series(lst_days_sequence)

# Methods for days in sequence strategy
def get_trading_day(df_asset, direction_col_name = 'day_direction', direction = 1, sequence_col_name = 'day_sequence', sequence_size = 3, zscore_col_name = 'z_score', th_zscore = 0.0):
    """
    Method to get the trading day of a sequence of events.
    
    
    Keyword arguments:
    df_asset (pandas.DataFrame) -- Pandas DataFrame with the asset data
    direction_col_name (String) -- Name of the column that have the information about the direction of the sequence
    direction (Integer) -- Number that represent the direction (0 = Short, 1 = Long)
    sequence_col_name (String) -- Name of the column that have the information about sequence of events
    sequence_size (Integer) -- Number that represent the sequence of events
    zscore_col_name (String) -- Name of the column that have the information about the z-score
    th_zscore (Integer) -- Number that represent the z-score
    """
    if th_zscore == 0.0:
        indexes = df_asset[(df_asset[direction_col_name] == direction) & (df_asset[sequence_col_name] == sequence_size)].index
        
        return indexes
    
    else:
        zscore = abs(th_zscore)
        if direction == 1:
            indexes = df_asset[(df_asset[direction_col_name] == direction) & (df_asset[sequence_col_name] == sequence_size) & (df_asset[zscore_col_name] >= zscore)].index
            return indexes
        
        elif direction == 0:
            indexes = df_asset[(df_asset[direction_col_name] == direction) & (df_asset[sequence_col_name] == sequence_size) & (df_asset[zscore_col_name] <= zscore*-1)].index
            return indexes


def get_trades(df_asset, trading_days_indexes, direction_col_name = 'day_direction', time_col_name = 'Date', price_col_name = 'Adj Close'):
        """
        Method for get trade data of days in sequence strategy

        Keyword arguments:
        df_asset (pandas.DataFrame) -- Pandas DataFrame with the asset data
        trading_days_indexes (Int64Index) -- Indexes of trading days
        direction_col_name (String) -- Name of the column that have the information about the direction of the sequence
        time_col_name (String) -- Name of the column that have the information about date time
        price_col_name (String) -- Name of the column that have the information about price
        """
        
        dict_trades = {}
        direction = df_asset.iloc[trading_day_indexes][direction_col_name].unique()[0]
        
        for target in [1,2,3,4,5,6,7,8,9,10]:
            lst_ops = []
            
            
            if direction == 1:
                op_type = 'short'
                for i in trading_days_indexes:
                    if i+target <= df_asset.index.max():
                        dt_sell = df_asset.loc[i, time_col_name]
                        price_sell = df_asset.loc[i, price_col_name]
                        dt_buy = df_asset.loc[i+target, time_col_name]
                        price_buy = df_asset.loc[i+target, price_col_name]
                        lst_ops.append([op_type, dt_buy, dt_sell, price_buy, price_sell])

            elif direction == 0:
                op_type = 'long'
                for i in trading_days_indexes:
                    if i+target <= df_asset.index.max():
                        dt_sell = df_asset.loc[i+target, time_col_name]
                        price_sell = df_asset.loc[i+target, price_col_name]
                        dt_buy = df_asset.loc[i, time_col_name]
                        price_buy = df_asset.loc[i, price_col_name]
                        lst_ops.append([op_type, dt_buy, dt_sell, price_buy, price_sell])
            
            dict_trades[str(target)] = lst_ops
        
        return dict_trades


def int_to_strperc(lst_values):
    for i in range(len(lst_values)):
        lst_values[i] = str(round((lst_values[i])*100, 2)) + '%'
    return lst_values


def get_strategy_results(dict_trades):
        """
        Method for get the results of the trades for different targets

        Keyword arguments:
        dict_trades (Dictionary) -- Dictionary with all trade data from get_trades method
        """

        dict_results = {'metrics': ['Média +', 'Maior', 'Média -', 'Menor', 'Taxa % +', 'Taxa % -', 'Exp. Mat.']}
        
        for target in dict_trades.keys():   
            df_res = pd.DataFrame(columns = ['op_type', 'dt_buy', 'dt_sell', 'price_buy', 'price_sell'], data = dict_trades[target])
            df_res['result'] = (df_res['price_sell'] - df_res['price_buy'])/df_res['price_buy']
            m_pos = df_res[df_res['result'] > 0]['result'].mean()
            max_pos = df_res[df_res['result'] > 0]['result'].max()
            m_neg = df_res[df_res['result'] < 0]['result'].mean()
            min_neg = df_res[df_res['result'] < 0]['result'].min()
            count_pos = df_res[df_res['result'] > 0].shape[0]
            count_neg = df_res[df_res['result'] < 0].shape[0]
            taxa_pos = (count_pos/(count_pos+count_neg))
            taxa_neg = (count_neg/(count_pos+count_neg))
            em = (taxa_pos * m_pos)+(taxa_neg * m_neg)
            lst_stats = [m_pos, max_pos, m_neg, min_neg, taxa_pos, taxa_neg, em]
            lst_stats_str = int_to_strperc(lst_stats)
            dict_results[target] = lst_stats_str
        
        print('Quantidade de ocorrências: ', (count_pos+count_neg))
        return pd.DataFrame(dict_results)


# Execution
asset = "itsa4.SA"
period = 20
high_low_dir = 0
seq_size = 5
zs_threshold = 0.0

df = yf.download(asset, period="max").reset_index().sort_values(by= 'Date')

df['ma'] = moving_average(df['Adj Close'], periods = period, ma_type = 'sma')
df['z_score'] = z_score(df['ma'], df['Adj Close'], moving_std_periods = period)
df['day_direction'] = set_high_low(df['Adj Close'])
df['day_sequence'] = days_in_sequence(df['day_direction'])

trading_day_indexes = get_trading_day(df, direction = high_low_dir, sequence_size = seq_size, th_zscore = zs_threshold)
dict_trades = get_trades(df, trading_day_indexes)

resultado = get_strategy_results(dict_trades)
best_target = resultado.iloc[6,1:].apply(lambda x: float(x[:-1])).idxmax()
msg_best_target = 'Melhor alvo: '+ best_target + ' dias.'

print(msg_best_target)
print(resultado)
