# Shiba Inu-Price-Prediction
Shiba Inu Coin Price Prediction with Machine Learning
Shiba Inu Coin is the reason for the recent drop in bitcoin prices. The price of Shiba Inu coin is currently very cheap compared to bitcoin, but some financial experts, claiming that we will see a rise in the price of Shiba Inu Coin soon. So, if you want to learn how to predict the future prices of Shiba Inu Coin. In this project, I will walk you through the task of Shiba Inu Price Prediction with Machine Learning using Python.
Shiba Inu Price Prediction
Predicting the price of a cryptocurrency is a regression problem in machine learning. Bitcoin is one of the most successful examples of cryptocurrency, but we recently saw a major drop in bitcoin prices due to shiba inu. Unlike bitcoin, dogecoin is very cheap right now, but financial experts are predicting that we may see a major increase in shiba inu prices.
There are many machine learning approaches that we can use for the task of Shiba Inu price prediction. You can train a machine learning model or you can also use an already available powerful model like the Facebook Prophet Model. But in the section below, I will be using the autots library in Python for the task of Shiba Inu coin price prediction with machine learning.
Shiba Inu Price Prediction using Python
To predict future Shiba Inu coin prices, you first need to get a dataset for this task. So the dataset for the Shiba Inu coin price are here
Now let’s get started with the task of Shiba Inu coin price prediction by importing the necessary Python libraries and the dataset.

Import Important Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
%matplotlib inline

Data Collection

data = pd.read_csv("SHIB-USD.csv")
data.head()

Data Pre-processing

data.info
data.info()
data.describe()
data.columns
data.isnull()
data.isnull().sum()

as we can see there is no null or missing value let's go for next step
In this dataset, the “close” column contains the values whose future values that we want to predict, so let’s have a closer look at the historical values of close prices of Shiba Inu coin

Feature Selection

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("Shiba Inu Price")
plt.xlabel("date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()

Now I will be using the autots library in Python to train a machine learning model for predicting the future prices of Dogecoin. If you have never used this library before then you can easily install it in your system by using the pip command

pip install autots

Now let’s train the Dogecoin price prediction model and have a look at the future prices of Shiba Inu coin
Training a machine learning model
from seaborn import regression

from autots import AutoTSmodel = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()forecast = prediction.forecastprint("Shiba Inu Price Prediction")print(forecast)

Summary
There are many machine learning approaches that we can use for the task of predicting the future prices of Shiba Inu coin. In this project, I introduced you to how you can predict the future prices of Shiba Inu coin by using the autots library in Python. I hope you like on how to predict the future prices of  Shiba Inu coin with Machine Learning using Python.
