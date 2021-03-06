# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:41:45 2015

@author: naman
"""

import scipy.io
from sklearn import svm,grid_search,decomposition
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import preprocessing

mat = scipy.io.loadmat('cs6923Project.mat')

test=mat.get('test');
train=mat.get('train');
train_label=mat.get('train_label');
train_label=(np.array(train_label)).ravel();
train=np.array(train);
test=np.array(test);

svc=svm.SVC()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])

scale1 = preprocessing.StandardScaler().fit(train)
train_scaled=scale1.transform(train)
test_scaled=scale1.transform(test)

param_grid = [ 
  {'svc__C': [0.01,0.001,0.1,1,10], 'svc__kernel': ['linear'],'pca__n_components':[20,25,35,40,45,50,70]},
  {'svc__C': [0.01, 0.1,1,10,0.001], 'gamma': [0.01,0.05,0.1,0.005,1,10], 'svc__kernel': ['rbf'],'pca__n_components':[20,25,35,40,45,50,70]},
{'svc__C': [0.01, 0.1, 1, 10], 'svc__epsilon': [1, 0.5, 0.1, 0.05], 'svc__degree' :[2, 3, 4], 'svc__kernel': ['poly'],'pca__n_components':[20,25,35,40,45,50,70]}]
print('Start') 
#svr = linear_model.LogisticRegression()
svrsearch= grid_search.GridSearchCV(pipe,param_grid,cv=10,n_jobs=10)
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
print (y_rbf.best_estimator_)
print(y_rbf.best_score_)
print(y_rbf.best_params_)

scipy.io.savemat('TrainPredict.mat',{'predict':predict})
scipy.io.savemat('TestPredict.mat',{'predict':predict1})

