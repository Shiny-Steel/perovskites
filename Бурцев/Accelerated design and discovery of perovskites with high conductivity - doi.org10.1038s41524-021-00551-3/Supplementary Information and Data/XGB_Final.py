import sys
import importlib

pyversion = sys.version.split()[0]
print('python', pyversion)

modules = ['xgboost','numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'cython',
               'mpmath', 'numexpr', 'sympy', 'virtualenv', 'mpi4py', 'sympy', 
               'numba', 'dask', 'networkx', 'skimage', 'pillow',
               'sqlalchemy', 'seaborn', 'bokeh', 'h5py', 'netCDF4',
	       'cffi', 'ctypes', 'ipyparallel']

for modulename in modules:
    try:
        imported_mod = importlib.import_module(modulename)
        print(modulename)	#		 imported_mod.__version__)
    except ImportError:
        print('{} not found'.format(modulename))

import numpy as npi
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics  
import xgboost as xgb
dataset=pd.read_csv("Conductivity_All_NNew.csv")

#For Regression
X=dataset[['MW','WPA','BM','CV','RHO','EMAX','EMIN','EDIF','RMAX','RMIN','RDIF','MV','POL','PERM','TEMP','PO2','PH2O','OV','AM_A_MIN','AM_A_MAX','AM_A_DIF','AM_A_AVG','AM_B_MIN','AM_B_MAX','AM_B_DIF','AM_B_AVG','BG_A_MIN','BG_A_MAX','BG_A_DIF','BG_A_AVG','BG_B_MIN','BG_B_MAX','BG_B_DIF','BG_B_AVG','BM_A_MIN','BM_A_MAX','BM_A_DIF','BM_A_AVG','BM_B_MIN','BM_B_MAX','BM_B_DIF','BM_B_AVG','CV_A_MIN','CV_A_MAX','CV_A_DIF','CV_A_AVG','CV_B_MIN','CV_B_MAX','CV_B_DIF','CV_B_AVG','RO_A_MIN','RO_A_MAX','RO_A_DIF','RO_A_AVG','RO_B_MIN','RO_B_MAX','RO_B_DIF','RO_B_AVG','EN_A_MIN','EN_A_MAX','EN_A_DIF','EN_A_AVG','EN_B_MIN','EN_B_MAX','EN_B_DIF','EN_B_AVG','FE_A_MIN','FE_A_MAX','FE_A_DIF','FE_A_AVG','FE_B_MIN','FE_B_MAX','FE_B_DIF','FE_B_AVG','IR_A_MIN','IR_A_MAX','IR_A_DIF','IR_A_AVG','IR_B_MIN','IR_B_MAX','IR_B_DIF','IR_B_AVG','MM_A_MIN','MM_A_MAX','MM_A_DIF','MM_A_AVG','MM_B_MIN','MM_B_MAX','MM_B_DIF','MM_B_AVG','MV_A_MIN','MV_A_MAX','MV_A_DIF','MV_A_AVG','MV_B_MIN','MV_B_MAX','MV_B_DIF','MV_B_AVG','OV_A','OV_B','POL_A_MIN','POL_A_MAX','POL_A_DIF','POL_A_AVG','POL_B_MIN','POL_B_MAX','POL_B_DIF','POL_B_AVG','TF','PERM_A','PERM_B','RRAT']]
y=dataset['COND_LOG']


#For Classification:
#X=dataset[['COND','MW','WPA','BM','CV','RHO','EMAX','EMIN','EDIF','RMAX','RMIN','RDIF','MV','POL','PERM','TEMP','PO2','PH2O','OV','AM_A_MIN','AM_A_MAX','AM_A_DIF','AM_A_AVG','AM_B_MIN','AM_B_MAX','AM_B_DIF','AM_B_AVG','BG_A_MIN','BG_A_MAX','BG_A_DIF','BG_A_AVG','BG_B_MIN','BG_B_MAX','BG_B_DIF','BG_B_AVG','BM_A_MIN','BM_A_MAX','BM_A_DIF','BM_A_AVG','BM_B_MIN','BM_B_MAX','BM_B_DIF','BM_B_AVG','CV_A_MIN','CV_A_MAX','CV_A_DIF','CV_A_AVG','CV_B_MIN','CV_B_MAX','CV_B_DIF','CV_B_AVG','RO_A_MIN','RO_A_MAX','RO_A_DIF','RO_A_AVG','RO_B_MIN','RO_B_MAX','RO_B_DIF','RO_B_AVG','EN_A_MIN','EN_A_MAX','EN_A_DIF','EN_A_AVG','EN_B_MIN','EN_B_MAX','EN_B_DIF','EN_B_AVG','FE_A_MIN','FE_A_MAX','FE_A_DIF','FE_A_AVG','FE_B_MIN','FE_B_MAX','FE_B_DIF','FE_B_AVG','IR_A_MIN','IR_A_MAX','IR_A_DIF','IR_A_AVG','IR_B_MIN','IR_B_MAX','IR_B_DIF','IR_B_AVG','MM_A_MIN','MM_A_MAX','MM_A_DIF','MM_A_AVG','MM_B_MIN','MM_B_MAX','MM_B_DIF','MM_B_AVG','MV_A_MIN','MV_A_MAX','MV_A_DIF','MV_A_AVG','MV_B_MIN','MV_B_MAX','MV_B_DIF','MV_B_AVG','OV_A','OV_B','POL_A_MIN','POL_A_MAX','POL_A_DIF','POL_A_AVG','POL_B_MIN','POL_B_MAX','POL_B_DIF','POL_B_AVG','TF','PERM_A','PERM_B','RRAT']]
#y=dataset['CLASS']

#Splitting the data randomly for testing(10%) validation (5%)
X1, XXV, Y1, YYV = train_test_split(X, y, test_size=0.05, random_state=42) 
XX,XXT,YY,YYT= train_test_split(X1, Y1, test_size=0.10, random_state=42)

#Fitting
regressor=xgb.XGBRegressor(n_estimators=150,max_depth=40)
#regressor=xgb.XGBClassifier(n_estimators=150,max_depth=40)
regressor.fit(XX, YY,eval_set=[(XXT, YYT)])

#Predictions
y_pred=regressor.predict(XX)
y_val=regressor.predict(XXV)
y_test=regressor.predict(XXT)


print('Validation set')
kk=-1
for items in YYV:
        kk=kk+1
        print(items,y_val[kk])

print('Feature Importance')
importances = list(zip(regressor.feature_importances_, X.columns))
for items in importances:
        print(items[0],items[1])

print('Testing set')
kk=-1
for items in YYT:
        kk=kk+1
        print(items,y_test[kk])

print('Root Mean Squared Error for test set:', npi.sqrt(metrics.mean_squared_error(y_test, YYT)))  
print('Root Mean Squared Error for validation set:', npi.sqrt(metrics.mean_squared_error(y_val, YYV)))
 
print('R2 for training:', r2_score(YY,y_pred))
print('R2 for testing:', r2_score(YYT,y_test))
print('R2 for validation:', r2_score(YYV,y_val))

