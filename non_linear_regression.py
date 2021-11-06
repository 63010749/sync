import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import model_selection
from sklearn import linear_model

# Import the data
weather_data = pd.read_csv('data1.csv',parse_dates=['datetime']
                     , infer_datetime_format=True)

# Check the shape of the dataset
print(weather_data.shape)

# Select the datetime and the temperature columns
temp_df = weather_data[["datetime","T_mu"]]
temp_df.head(10)

# Select the subset data from 2016 to 2019
mask = (temp_df['datetime'] >= '01/01/2018') & (temp_df['datetime'] <= '31/01/2018')
temp_df = temp_df.loc[mask]

# Reset the index 
temp_df.set_index("datetime", inplace=True)

# Inspect first 5 rows and last 5 rows of the data
from IPython.display import display
display(temp_df.head(5))
display(temp_df.tail(5))

y = [e for e in temp_df.T_mu]
x = list(range(1,len(y)+1))
df = pd.DataFrame(
    {'x': x,
     'y': y}
)

# Show the first five rows of our dataframe
df.head()


# Calculate the mean of x and y
xmean = np.mean(x)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of beta

xy= (df['x']  * df['y']).sum() 
sx= x.sum()
sy= df['y'].sum()
x2y= ((df['x']**2)*df['y']).sum()
x2= (df['x']**2).sum()
x3= (df['x']**3).sum()
x4= (df['x']**4).sum()
print("{} {} {} {} {} {} {}".format(xy,sx,sy,x2y,x2,x3,x4))





