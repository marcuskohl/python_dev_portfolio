import pandas as pd
import ta
from ta import add_all_ta_features

def create_lag_features(df, lags):
    """Creating lag features for time series for multiple columns."""
    lagged_dfs = []
    for column in ['open', 'high', 'low', 'close', 'volume']:
        for lag in lags:
            lagged_df = df[[column]].shift(lag).rename(columns={column: f'{column}_lag_{lag}'})
            lagged_dfs.append(lagged_df)
    df_lagged = pd.concat(lagged_dfs, axis=1)
    return pd.concat([df, df_lagged], axis=1)

def add_day_of_week(df):
    """Adding day of week as a feature."""
    day_of_week = df.index.dayofweek.to_frame(name='day_of_week')
    day_of_week.index = df.index 
    return pd.concat([df, day_of_week], axis=1)

def main():
    #Loading data
    data_path = '../data/BTCUSDT_Hourly_Cleaned.csv'
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    
    #Adding TA features
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    
    #Creating lag features
    lags = [1, 3, 7, 14, 30]
    df = create_lag_features(df, lags)

    #Adding day of week feature
    df = add_day_of_week(df)

    #Dropping first 30 rows to avoid NaN values from lag features
    df = df.iloc[30:]

    #Saving processed data
    output_path = '../data/processed_BTCUSDT_Hourly.csv'
    df.to_csv(output_path)

    print(f"Processed data saved to {output_path}")

if __name__ == '__main__':
    main()



