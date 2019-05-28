#SVR comparison for different test size splits (Bar Graph)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv("day_new.csv")
data.head()

dataset = data[['season','yr','mnth','weekday','holiday','weathersit','d_atemp','d_hum','d_windspeed','cnt']]

#dataset.head() 
x = dataset.iloc[:, 0:9].values  
y = dataset.iloc[:, 9].values

r=[]
t=[0.1,0.2,0.3,0.4,0.5] #different test sizes
for i in t:
    
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=i)

    from sklearn.svm import SVR  #import svr

    svr_lin = SVR(kernel='linear', C=100, gamma=0.4)
    svr_lin.fit(x_train,y_train)
    y_pred=svr_lin.predict(x_test)

    from sklearn.metrics import r2_score
    print(r2_score(y_test , y_pred))
    r.append(r2_score(y_test , y_pred))
    

import matplotlib.pyplot as plt
import numpy as np
n_groups=5
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
#drawing bar graph  
rects1 = plt.bar(index, r, bar_width,alpha=opacity,color='b')
  
plt.xlabel('Test Size')
plt.ylabel('r2_Score')
plt.title('Model Evaluation with different test size')
plt.xticks(index,('10%', '20%', '30%', '40%', '50%'))
plt.legend()
 
plt.tight_layout()
plt.show()
