# Stock-price-prediction for APPLE

## I. DEFINE PROBLEM

Machine learning and deep learning have been transforming finance and investement industry. AI powered trading coulf potentially reduce the risk and maximize returns. So the goal of the project is to leverage a model to predict the future stock prices. By accurately predicting stock prices, investors can maximize returns and can get an idea as to when they should buy or sell securities. The AI/ML model will be trained using a type of recurrent neural network(RNN)know as long short term memory networks(LSTM).Investment firms have adopted machine learning in recent years rapidly, even some firms have started replacing humans with A.I. to make investment decisions.In this project, 100 days stock closing values is used to predict the next day's stock price. 

**Environment or libraries used**

* Google Colab
* Python 3
* Numpy
* Pandas
* Sklearn
* Matplotlib
* Keras

**Datasets**

./data directory contains .csv files

![stock_price_data.png](https://github.com/Jimisha18/Stock-price-forecasting-for-APPLE/blob/main/IMAGES/stock_price_data.png)


## II. DISCOVER
**Exploratory Data analysis**

```python
# Function to plot interactive plots 
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()

```

![stock price_EDA.png](https://github.com/Jimisha18/Stock-price-forecasting-for-APPLE/blob/main/IMAGES/Stock%20price_EDA.png)

**Normalization**

```
# Function to normalize stock prices based on their initial price
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x
```

## III. 
**Metrics**

To determine how accurate the prediction is, we analyze the difference between the predicted and the actual adjusted close price. Smaller the difference indicates better accuracy.

I chose both Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) as a metric to determine the accuracy of the prediction. It is a commonly used general purpose quality estimator.

Also, by visualizing the predicted price and the actual price with a plot or a graph, it can tell how close the prediction is clearly.

Why I use MSE/RMSE for the metric?

There are many metrics for accuracy like R2, MAE, etc.

I chose to use MSE/RMSE because they explicitly show the deviation of the prediction for continuous variables from the actual dataset. So, they fit in this project to measure the accuracy.


![](IMAGES/rmse.gif)

It measures the average magnitude of the error and ranges from 0 to infinity. The errors are squared and then they are averaged, MSE/RMSE gives a relatively high weight to large errors, and the errors in stock price prediction can be critical, so it is appropriate metric to penalize the large errors.



