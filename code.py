### Loading modules
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
 
dataset = pd.read_csv("D:\ML\Student admission problem\Admission_Predict_Ver1.1.csv")

x = dataset.iloc[:,1:8]
y = dataset.iloc[:,8]

##PLOT

gre_plot=plt.scatter(x.iloc[:,1],y)
gre_plot=plt.xlabel("GRE Score")
gre_plot=plt.ylabel("Chance")

## Data descr

dataset.info()  ### no null values

## for detecting outliers

dataset.boxplot(column = ["GRE Score","TOEFL Score","University Rating","SOP","CGPA"])  ## no outliers were detected

# For normalization we would use sklearn modules... there are many sub normalisation techniques
from sklearn.preprocessing import minmax_scale

x = minmax_scale(x, feature_range = (0,1))

### Splitting the data into train and test sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,train_size=0.7,random_state=37)

### making data ready for multiple regression
from sklearn.preprocessing import MinMaxScaler

Sc = MinMaxScaler(feature_range=(0,1))


x_train = Sc.fit_transform(x_train)

x_test = Sc.fit_transform(x_test)

## We wont transform or reshape Y as its already in the range of 0-1

"""
Multiple linear regression

"""
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

predictlr = lr.predict(x_test)

### Evaluation of this model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def mean_absolute_percentage_error(y_true,y_predict):
    y_true,y_predict = np.array(y_true),np.array(y_predict)
    return np.mean(np.abs((y_true-y_predict)/y_true))*100

mae_lr = mean_absolute_error(y_test, predictlr)
mse_lr = mean_squared_error(y_test, predictlr)
rs_lr = r2_score(y_test, predictlr) ## 78%
rmse = math.sqrt(mse_lr)
mape_lr = mean_absolute_percentage_error(y_test,predictlr)

### This model has error of 6.7% on mape


"""""
Polynomial linear regression

"""""
from sklearn.preprocessing import PolynomialFeatures

## As the data is already transformed we dont need to transform it again

## We will transform the data in poly form so that we can apply this model

poly_p = PolynomialFeatures(degree=2)

poly_x = poly_p.fit_transform(x_train)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

poly_lr = lr.fit(poly_x,y_train)

## now reshaping the test set for getting the outcome

poly_x_test= poly_p.fit_transform(x_test)

predict_plr = poly_lr.predict(poly_x_test)

### Evaluation of this model

mae_plr = mean_absolute_error(y_test, predict_plr)
mse_plr = mean_squared_error(y_test, predict_plr)
rs_plr = r2_score(y_test, predict_plr) ### 76%
rmse = math.sqrt(mse_plr)
mape_plr = mean_absolute_percentage_error(y_test,predict_plr)

### best degree is 2 for this model, as its mostly linear kind of data

### This model has error of 6.8% on mape


""""
Random forest model

""""
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=33)

rf.fit(x_train,y_train)

predictrf=rf.predict(x_test)

predictrf = predictrf.reshape(-1,1)

### Evaluation of this model

mae_rfm = mean_absolute_error(y_test, predictrf)
mse_rfm = mean_squared_error(y_test, predictrf)
rs_rfm = r2_score(y_test, predictrf) ## 64 %
rmse_rfm = math.sqrt(mse_rfm)
mape_rfm = mean_absolute_percentage_error(y_test,predictrf)

### This model has error of 21% on mape


""""
Support vector regression

""""
from sklearn.svm import SVR

svr = SVR()
svr.fit(x_train,y_train)

predictsvr = svr.predict(x_test)

## Evaluation of this model

mae_svr = mean_absolute_error(y_test, predictsvr) 
mse_svr = mean_squared_error(y_test, predictsvr)  
rmse_svr = math.sqrt(mse_svr) 
rs_svr = r2_score(y_test, predictsvr) ## 64.2 %
mape_svr= mean_absolute_percentage_error(y_test,predictsvr) #9%

### This model has error of 9% on mape




