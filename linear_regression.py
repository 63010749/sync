import sklearn
import numpy as np
import pandas as pd
import matplotlib
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
weather_data = pd.read_csv('North.csv',parse_dates=['datetime']
                     , infer_datetime_format=True)

# Check the shape of the dataset
#print(weather_data)
temp_t = weather_data["T_mu"]
print(temp_t)
# Select the datetime and the temperature columns
temp_df = weather_data[["datetime","T_mu"]]
temp_df.head(10)

# Select the subset data from 2021 to 2021
mask = (temp_df['datetime'] >= '01/01/2021') & (temp_df['datetime'] <= '31/10/2021')
temp_df = temp_df.loc[mask]

# Reset the index 
temp_df.set_index("datetime", inplace=True)

# Inspect first 5 rows and last 5 rows of the data
from IPython.display import display

y = [e for e in temp_df.T_mu]
y2 = [e for e in temp_df.]
print(y)
x = list(range(1,len(y)+1))   #[f for f in temp_df.datetime]
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
df['xycov'] = (df['x'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['x'] - xmean)**2

# Calculate beta and alpha
beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
print(f'alpha = {alpha:.3f}')
print(f'beta = {beta:.3f}')
print(f'y(predict) = {beta:.3f}x + {alpha:.3f}')

ypred = [alpha + beta*e for e in x]

corr_matrix = np.corrcoef(y, ypred)
corr = corr_matrix[0,1]
R_sq = corr**2
print(f'R-square = {R_sq:.3f}')

MSE = np.square(np.subtract(y,ypred)).mean()
print(f'MSE = {MSE:.3f}')
yText=min(y)
# Plot regression against actual data
fig = plt.figure(figsize=(12, 6))
#ax = fig.add_subplot()
plt.plot(x, y)# scatter plot showing actual data
plt.plot(x, ypred)# regression line   
plt.title('North 2021')
plt.xlabel('days')
plt.ylabel('temperature(C)')
plt.annotate(f'y(predict) = {beta:.3f}x + {alpha:.3f}',(270,yText+2))
plt.annotate(f'R-square = {R_sq:.3f}', (270, yText+1))
plt.annotate(f'MSE = {MSE:.3f}',(270,yText))


#plt.show()



