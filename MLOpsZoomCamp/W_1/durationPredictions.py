#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import mlflow

mlflow.set_tracking_uri("sqlite:///durationPred.db")
mlflow.set_experiment("DurationPredictions-experiment")
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_extraction import DictVectorizer
from typing import List, Dict

#%%    
train_filename : str  ="green_tripdata_2021-01.parquet"
validation_filename : str= "green_tripdata_2021-02.parquet"

categorical : List[str] = ["PU_DO"]
numerical : List[str] = ["trip_distance"]
def readShit():
    df = pd.read_parquet(train_filename)
    df_test = pd.read_parquet(validation_filename)
    return df, df_test



def prepareDataset(df : pd.DataFrame) -> pd.DataFrame:
    convertColumns : List[str] = ["lpep_pickup_datetime", "lpep_dropoff_datetime"] 
    df[convertColumns[0]] = pd.to_datetime(df[convertColumns[0]])
    df[convertColumns[1]] = pd.to_datetime(df[convertColumns[1]])

    df["PU_DO"] = df["PULocationID"].astype(str) +"_" + df["DOLocationID"].astype(str)
    
    df['duration'] = df[convertColumns[1]] - df[convertColumns[0]]
    df['duration'] = df["duration"].apply(lambda row : row.total_seconds() / 60)

    # Filterisasi, sehingga hanya mendapatkan data untuk perjalanan
    # >1 sampai satu jam saja

    df = df[( (df["duration"] >= 1) & (df["duration"] <=60))]

    return df

#%%
df,df_test = readShit()
df = prepareDataset(df)
df_test =prepareDataset(df_test)
print(df)

#%%
#%%
def fitLinearModel(x_train, y_train, x_test, y_test,type: str):
    linReg = ""
    if(type == "linreg"):
        linReg = LinearRegression()
    elif(type == "lasso"):
        linReg = Lasso()
    else:
        linReg = Ridge()
    linReg.fit(x_train, y_train)

    y_pred =linReg.predict(x_test)
    sns.distplot(y_pred,bins=10, label=f"Prediction using {type} Regression")
    sns.distplot(y_test,bins=10, label="True Label")
    mlFlowMetricParamRequirements(mean_squared_error(y_test,y_pred,squared=False))
    plt.title(f"{type} Regression RMSE %d" %(mean_squared_error(y_test,y_pred,squared=False)))
    plt.legend()
    plt.show()
    if(type == "linReg"):
        mlFlowLinearRegressionLogParams(linReg.coef_, linReg.intercept_)
    elif(type == "lasso"):
        mlFlowLassoLogParams(linReg.coef_, linReg.intercept_, 1)
    else:
        mlFlowRidgeLogParams(linReg.coef_, linReg.intercept_, 1)
    
    return linReg
def mlFlowGeneralParamRequirements():

    mlflow.set_tag("Developer", "Primitif")
    mlflow.log_param("train-data-filename", train_filename)
    mlflow.log_param("validation-data-filename", validation_filename)
    mlflow.log_param("used_categorical_columns", categorical)
    mlflow.log_param("used_numerical_columns", numerical)

def mlFlowMetricParamRequirements(rmse):
    metric = {"RMSE" : rmse}
    mlflow.log_metrics(metric)

def mlFlowLinearRegressionLogParams(coeffs, intercept):
    params = {
        "coeffs" : coeffs,
        "intercept" : intercept,
    }
    mlflow.log_params(params)

def mlFlowRidgeLogParams(coeffs, intercept, alpha):

    params = {
        "coeffs" : coeffs,
        "intercept" : intercept,
    }
    mlflow.log_params(params)

def mlFlowLassoLogParams(coeffs, intercept, alpha):
    params = {
        "coeffs" : coeffs,
        "intercept" : intercept,
        "alpha" : alpha,
    }
    mlflow.log_params(params)

# %%
with mlflow.start_run():
    mlFlowGeneralParamRequirements()
    train_dicts = df[categorical + numerical].to_dict(orient="records")
    test_dicts = df_test[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    target : str = "duration"
    Y_train = df[target].values
    Y_test = df_test[target].values
    fitLinearModel(X_train, Y_train, X_test, Y_test,"linreg" )
#%%