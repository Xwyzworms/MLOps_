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
    with open("models/linReg.bin", "wb") as fuf:
        pickle.dump((dv, linReg), fuf)
    
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
    mlflow.log_artifact(local_path="models/linReg.bin", artifact_path="modelsPreprocessor")
#%%
#### ML Flow autoLog using xgboost
import xgboost as xgb   
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

train_xgb = xgb.DMatrix(X_train, label=Y_train)
test_xgb = xgb.DMatrix(X_test, Y_test)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost_v1")
        mlflow.xgboost.autolog()

        booster = xgb.train(params=params,
                             dtrain=train_xgb,
                             num_boost_round=100,
                             evals=[(test_xgb, "validation")],
                             early_stopping_rounds=50
                             )
        yPred = booster.predict(test_xgb)
        rmse = mean_squared_error(Y_test, yPred, squared=False)
        mlflow.log_metric("RMSE", rmse)

    return {"loss" : rmse, "status" : STATUS_OK}

search_space = {
    "max_depth" : scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate" : hp.loguniform("learning_rate", -3, 0),
    "reg_alpha" : hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda" : hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", 1,3),
    "objective" : "reg:linear",
    "seed" : 42,
} 
#%%
xgboostResult = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials= Trials()
)

#%% 
## Using best parameters 
params = {
    "custom_metric":	None,
    "early_stopping_rounds" :	50,
    "learning_rate":	0.2615664281684017,
    "max_depth":	24,
    "maximize":	None,
    "min_child_weight":	18.714937722242695,
    "num_boost_round":	100,
    "objective":"reg:linear",
    "reg_alpha"	:0.055991915605540066,
    "reg_lambda":	0.037685837026633744,
    "seed"	:42,
    "verbose_eval"	:True,
}


booster = xgb.train(params=params,
                        dtrain=train_xgb,
                        num_boost_round=100,
                        evals=[(test_xgb, "validation")],
                        early_stopping_rounds=50
                        )
yPred = booster.predict(test_xgb)
with open("models_best/1/dataPreprocessor.b","wb") as fuf:
    pickle.dump(dv, fuf)

rmse = mean_squared_error(Y_test, yPred, squared=False)
mlflow.log_metric("RMSE", rmse)
mlflow.xgboost.log_model(booster, artifact_path="modelPreprocessor")
mlflow.log_artifact(local_path="models_best/1/dataPreprocessor.b", artifact_path="dataPreprocessor")
# %%
