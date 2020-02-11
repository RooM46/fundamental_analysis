# Pandas in order to view and format the data
import pandas as pd

# Pandas option so we can see all rows and columns
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 70)

# Numpy for necessary calculations
import numpy as np

# Jupyter widget progress bar
from tqdm import tqdm_notebook as tqdm

# Pickle to import and export the formatted datasets
import _pickle as pickle

# Loading a pickle file that contains a dictionary with the keys being stock tickers and values being a DataFrame of QRs
with open("MSFT_quarterly_financial_data (1).pkl",'rb') as fp:
    stocks_df = pd.read_pickle('MSFT_quarterly_financial_data (1).pkl')

stocks_df = pd.read_pickle("MSFT_quarterly_financial_data (1).pkl")
print(stocks_df.keys())
def setting_index(stocks_df):
    """
    Returns a sorted datetime index
    """
    
    stocks_df['Quarter end'] = pd.to_datetime(stocks_df['Quarter end'])
    df.set_index("Quarter end", inplace=True)
    return df.sort_index(ascending=True)
  
def class_creation(df, thres=3):
    """
    Creates classes of:
    - buy(1)
    - hold(2)
    - sell(0)
    
    Threshold can be changed to fit whatever price percentage change is desired
    """
    if df['Price high'] >= thres and df['Price low'] >= thres:
        # Buys
        return 1
    
    elif df['Price high'] <= -thres and df['Price low'] <= -thres:
        # Sells
        return 0
    
    else:
        # Holds
        return 2
# Setting the index as the Date
for i in tqdm(stocks_df.keys()):
    stocks_df[i] = setting_index(stocks_df[i])
    
# Replacing all "None" values with NaN
for i in tqdm(stocks_df.keys()):
    stocks_df[i].replace("None", 0, inplace=True)
    
# Creating a new dictionary that contains the numerical values, then converting all values to numeric values
num_df = {}
for i in tqdm(stocks_df.keys()):
    num_df[i] = stocks_df[i].apply(pd.to_numeric)
    
# Replacing values with percent difference or change
pcnt_df = {}
for i in tqdm(num_df.keys()):
    pcnt_df[i] = num_df[i].pct_change(periods=1).apply(lambda x: x*100)
    
# Replacing infinite values with NaN
for i in tqdm(pcnt_df.keys()):
    pcnt_df[i] = pcnt_df[i].replace([np.inf, -np.inf], np.nan)
    
# Creating a new DataFrame that contains the class 'Decision' determining if a quarterly reports improvement is a buy, hold, or sell.
new_df = {}
for i in tqdm(pcnt_df.keys()):
    # Assigning the new DF
    new_df[i] = pcnt_df[i]
    
    # Creating the new column with the classes, shifted by -1 in order to know if the prices will increase/decrease in the next quarter.
    new_df[i]['Decision'] = new_df[i].apply(class_creation, axis=1).shift(-1)
    
# Excluding the first and last rows
for i in tqdm(new_df.keys()):
    new_df[i] = new_df[i][1:-1]
    
# Combining all stock DFs into one
big_df = pd.DataFrame()
for i in tqdm(pcnt_df.keys()):
    big_df = big_df.append(new_df[i], sort=False)
    
# Filling the NaNs with 0
big_df.fillna(0, inplace=True)

# Resetting the index because we no longer need the dates
big_df.reset_index(drop=True, inplace=True)

# Dropping the price related columns to prevent data leakage
big_df.drop(['Price', 'Price high', 'Price low'], 1, inplace=True)

# Exporting the final DataFrame
with open("main_df.pkl", 'wb') as fp:
    pickle.dump(big_df, fp)