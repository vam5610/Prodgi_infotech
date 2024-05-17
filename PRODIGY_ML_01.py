import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


#TRAIN DATASET


td=pd.read_csv(r'D:\Testing\new\train.csv')
print(td.columns)
num_td=td.select_dtypes(include="number")
corr_matrix=num_td.corr()
corr_matrix["SalePrice"].sort_values(ascending = False)

#TAKING ONLY REQUIRED COLUMNS
req_traindata = ["GarageArea","OverallQual","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","TotRmsAbvGrd","SalePrice"]

#SELECTED ONLY REQUIRED COLUMNS 
selected_data=td[req_traindata]
selected_data.loc[:,'Totalbath']=(selected_data['BsmtFullBath'].fillna(0)+
                                 selected_data['BsmtHalfBath'].fillna(0)+
                                 selected_data['FullBath'].fillna(0)+
                                 selected_data['HalfBath'].fillna(0))

selected_data.loc[:,'Totalarea']=(selected_data['GarageArea'].fillna(0)+
                                  selected_data['OverallQual'].fillna(0)+
                                  selected_data['TotalBsmtSF'].fillna(0)+
                                  selected_data['1stFlrSF'].fillna(0)+
                                  selected_data['2ndFlrSF'].fillna(0)+
                                  selected_data['GrLivArea'].fillna(0))
print(selected_data)
new_td = selected_data[['TotRmsAbvGrd','Totalbath','GarageArea','Totalarea','OverallQual','SalePrice']]
print(new_td)
train_set,test_set=train_test_split(new_td,test_size=0.2,random_state=42)
print(train_set.shape, test_set.shape)
housing=train_set.drop('SalePrice',axis=1)
house_labels= train_set['SalePrice'].copy()


#CREATING THE PIPELINE FOR DATA POINTS WHICH RANGES ARE DISTURBUTED WIDELY 

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler() )
])
X_train = my_pipeline.fit_transform(housing)
print(X_train)
Y_train=house_labels
print(Y_train.shape)

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sns.pairplot(new_td)
plt.tight_layout()
plt.show()
sns.heatmap(new_td.corr(),annot=True)




#TEST DATA PROCESSING 



testdf = pd.read_csv(r'D:\Testing\new\test.csv')
print(testdf.head)

req_tst = ["GarageArea","OverallQual","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","TotRmsAbvGrd"]
selected_tst = testdf[req_tst]
selected_tst.loc[:, 'TotalBath'] = (selected_tst['BsmtFullBath'].fillna(0) +
                                    selected_tst['BsmtHalfBath'].fillna(0) +
                                    selected_tst['FullBath'].fillna(0) +
                                    selected_tst['HalfBath'].fillna(0))

selected_tst.loc[:, 'TotalSF'] = (selected_tst['TotalBsmtSF'].fillna(0) +
                                  selected_tst['1stFlrSF'].fillna(0) +
                                  selected_tst['2ndFlrSF'].fillna(0) +
                                  selected_tst['LowQualFinSF'].fillna(0) +
                                  selected_tst['GrLivArea'].fillna(0))
print(selected_tst)
test_df_unproc = selected_tst[['TotRmsAbvGrd','TotalBath','GarageArea','TotalSF','OverallQual']]
print(test_df_unproc)
test_df = test_df_unproc.fillna(test_df_unproc.mean())
X_test = my_pipeline.transform(test_df[['TotRmsAbvGrd','TotalBath','GarageArea','TotalSF','OverallQual']].values)
print(X_test)


#SELECTING THE MODEL 
model = LinearRegression()

model.fit(X_train,Y_train)
y_train_pred = model.predict(X_train)
y_train_pred[:5]
some_data = housing.iloc[:5]
some_labels = house_labels.iloc[:5]
proc_data = my_pipeline.transform(some_data)
model.predict(proc_data)
print(list(some_labels))
train_mse = mean_squared_error(Y_train,y_train_pred)
train_rmse = np.sqrt(train_mse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X_train,Y_train,scoring="neg_mean_squared_error",cv = 200)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation",scores.std())

print_scores(rmse_scores)
y_pred=model.predict(X_test)
print(y_pred)










