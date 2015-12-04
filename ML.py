import scipy.io
from sklearn import svm,grid_search
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation

mat = scipy.io.loadmat('cs6923Project.mat')

test=mat.get('test');
train=mat.get('train');
train_label=mat.get('train_label');
train_label=(np.array(train_label)).ravel();
train=np.array(train);
test=np.array(test);
scale1 = preprocessing.StandardScaler().fit(train)
train_scaled=scale1.transform(train)
test_scaled=scale1.transform(test)

param_grid = [
  {'C': [0.01], 'kernel': ['linear']},
  {'C': [0.01, 0.1], 'gamma': [0.01], 'kernel': ['rbf']},

 ]
print('Start') 
svr = svm.SVC()
svrsearch= grid_search.GridSearchCV(svr,param_grid,cv=10,n_jobs=10)
y_rbf = svrsearch.fit(train_scaled, train_label)
print('Done')
predict=[]
predict1=[]
a=[]
j=0;
for i in range(50000):
    predict.append(y_rbf.predict(train_scaled[i]))
    predict1.append(y_rbf.predict(test_scaled[i]))
    if(y_rbf.predict(train_scaled[i])==train_label[i]):
        j=j+1;
predict=(np.array(predict)).ravel();
predict1=(np.array(predict1)).ravel();
print(j)

scipy.io.savemat('TrainPredict.mat',{'predict':predict})
scipy.io.savemat('TestPredict.mat',{'predict':predict1})
