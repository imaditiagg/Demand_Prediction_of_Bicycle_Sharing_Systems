#Plotting of Regressor Line on test-size 0.2 and calculating different accuracy measures
import pandas as pd
import numpy as np 
data=pd.read_csv("day_new.csv")
data.head()

dataset = data[['season','yr','mnth','weekday','holiday','weathersit','d_atemp','d_hum','d_windspeed','cnt']]

#dataset.head() 
x = dataset.iloc[:, 0:9].values  
y = dataset.iloc[:, 9].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)

from sklearn.svm import SVR
#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.2, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma=0.4)
#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)

svr_lin.fit(x_train,y_train)
y_pred=svr_lin.predict(x_test)

#printing of different accuracy measures
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
print(r2_score(y_test , y_pred))

#Plotting of Regressor Line
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


