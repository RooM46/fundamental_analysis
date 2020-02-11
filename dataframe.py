import pandas as pd

df = pd.read_excel (r'C:\Users\rache\OneDrive\Documents\dev\fundamental_analysis\MSFT_quarterly_financial_data (1).xlsm')

df.to_pickle('MSFT_quarterly_financial_data (1).pkl')

print(type(df.index))

print(df.keys())

def setting_index(df):
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    df.set_index("Quarter end", inplace=True)
    return df.sort_index(ascending=True)


