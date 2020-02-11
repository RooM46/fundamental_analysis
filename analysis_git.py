# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Cleaning and Formatting the Datasets
# Cleaning up the Data from each quarterly report with respect to its stock ticker.
# %% [markdown]
# ### Importing Libraries and Data

# %%
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 70)
import numpy as np
from tqdm import tqdm_notebook as tqdm
import _pickle as pickle

stocks_df = pd.read_pickle("MSFT_quarterly_financial_data (1).pkl")
print(type(stocks_df)

# %% [markdown]
# ## Preprocessing the Data
# %% [markdown]
# ### Setting the Index to the Date

# %%
def setting_index(df):
    """
    Returns a sorted datetime index
    """
    print(stock_df.keys)
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    df.set_index("Quarter end", inplace=True)
    return df.sort_index(ascending=True)

for i in tqdm(stocks_df.keys()):
    stocks_df[i] = setting_index(stocks_df[i])

# %% [markdown]
# ### Replacing all "None" values with zero

# %%
for i in tqdm(stocks_df.keys()):
    stocks_df[i].replace("None", 0, inplace=True)

# %% [markdown]
# ### Converting all values to numeric values

# %%
# Creating a new dictionary that contains the numerical values
num_df = {}

for i in tqdm(stocks_df.keys()):
    num_df[i] = stocks_df[i].apply(pd.to_numeric)

# %% [markdown]
# ### Replacing values with percent difference 
# (Between each quarter)
# 
# Also, mutliplying by 100 for better readability

# %%
pcnt_df = {}

for i in tqdm(num_df.keys()):
    pcnt_df[i] = num_df[i].pct_change(periods=1).apply(lambda x: x*100)

# %% [markdown]
# #### Replacing infinite values with NaN

# %%
for i in tqdm(pcnt_df.keys()):
    pcnt_df[i] = pcnt_df[i].replace([np.inf, -np.inf], np.nan)

# %% [markdown]
# ## Creating the Classes
# - Buy (because the highest high and lowest low of the quarter will both increase by 3% or more)
# - Sell (because the lowest low and highest high of the quarter will both decrease by 3% or more)
# - Hold (because it will not do either)

# %%
def class_creation(df, thres=3):
    """
    Creates classes of:
    - buy(1)
    - hold(2)
    - sell(0)
    
    Threshold can be changed to fit whatever price change is desired
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

# %% [markdown]
# Creating a new DataFrame that contains the class 'Decision' determining if a quarterly reports improvement is a buy, hold, or sell.

# %%
new_df = {}

for i in tqdm(pcnt_df.keys()):
    # Assigning the the new DF
    new_df[i] = pcnt_df[i]
    
    # Creating the new column with the classes, shifted by -1 in order to know if the prices will increase/decrease in the next quarter.
    new_df[i]['Decision'] = new_df[i].apply(class_creation, axis=1).shift(-1)

# %% [markdown]
# ### Excluding the first and last rows
# This is done because the last row has no data to compare percent improvements to and the first row does not have any data to show if the price will increase in the future.

# %%
for i in tqdm(new_df.keys()):
    new_df[i] = new_df[i][1:-1]

# %% [markdown]
# #### Examining an example DF to check if the classes were assigned correctly

# %%
new_df['A'][['Price high', 'Price low', 'Decision']]

# %% [markdown]
# ### Combining all stock DFs into one

# %%
big_df = pd.DataFrame()

for i in tqdm(pcnt_df.keys()):
    big_df = big_df.append(new_df[i], sort=False)

# %% [markdown]
# #### Quick check for NaN values

# %%
big_df.isna().sum()

# %% [markdown]
# #### Filling the NaNs with 0

# %%
big_df.fillna(0, inplace=True)


# %%
# Checking the DF again for NaN
big_df.isna().sum()

# %% [markdown]
# ### Resetting the index
# Because we no longer need the dates

# %%
big_df.reset_index(drop=True, inplace=True)
big_df.head()

# %% [markdown]
# ### Dropping the Prices columns
# - Price
# - Price high
# - Price low
# 
# To prevent any data leakage because we are looking mainly at the QR's value changes rather than prices.

# %%
big_df.drop(['Price', 'Price high', 'Price low'], 1, inplace=True)

# %% [markdown]
# ### Counting how many classes there are

# %%
big_df['Decision'].value_counts()

# %% [markdown]
# Unequal classes are fine because we will use a specific evaluation metric to determine success in classification.
# %% [markdown]
# ### Exporting the final dataframe

# %%
with open("main_df.pkl", 'wb') as fp:
    pickle.dump(big_df, fp)


# %%



# %%



# %%


