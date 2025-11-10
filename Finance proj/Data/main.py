import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#Data loading
balance_df = pd.read_excel('balance.xlsx')
features = ['Gear','Year of manufacture','modelyr','city']

#outliers removal using IQR
Q1 = balance_df.quantile(0.05)
Q3 = balance_df.quantile(0.95)
IQR = Q3 - Q1
balance_df = balance_df[~((balance_df < (Q1 - 1.5 * IQR)) | (balance_df > (Q3 + 1.5 * IQR))).any(axis = 1)]

#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
balance_df[features] = scaler.fit_transform(balance_df[features])

from sklearn.model_selection import train_test_split
X = balance_df[features]
Y = balance_df['price_in_lakhs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(f'Mean Squared Error: {mse}')
r2_score(y_test,y_pred)
len(X_train.columns)

importances = model.feature_importances_
feature_importances = pd.DataFrame({'Features': X.columns, 'Importance':importances})
feature_importances = feature_importances.sort_values(by = 'Importance', ascending=False)
feature_importances

parameters = {
    'n_estimators': [50,100,150,200,250,300,350,400], #More granularity for tree count
    'max_depth': [5,10,15,20,'None'], #Including deeper trees and no limit
    'min_samples_split': [2,5,10,20], #Explore stricter splitting rules
    'bootstrap': [True, False], #Test both bootstrapping & non-bootstrapping
    'criterion': ['squared_error','absolute_error','poisson'], #Include Poisson criterion
    'max_features': ['auto','sqrt','log2'], #explore feature subsets
    'oob_score': [True, False], #out of bag score for more robust validation
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=-1)
grid_search.fit(X_train,y_train)

best_params = grid_search.best_params_
print(best_params)

#Train model with best parameters
model = RandomForestRegressor(**best_params)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print('Mean Squared error': {mse})
r2_score(y_test, y_pred)

#model assessment
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2:{r2}')

#ML flow
#Run - mlflow ui - in terminal to open mlflow website

#import pickle
#pickle.dump(model, open('random_forest_model.pkl','wb'))

import mlflow
mlflow.set_experiment('first experiment')
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

#Fitting a single model

#Track all parameters in mlflow website
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    mlflow.sklearn.log_model(model,'Random Forest Regressor')


#Fitting Multiple models

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

models = [
    (
        'Gradient Boosting Regressor',

        {'n_estimators': 150, 'learning_rate': 0.1},
        GradientBoostingRegressor(), #example of hyperparameters
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        'Random Forest',

        {'n_estimators': 150, 'max_depth': 5},
        RandomForestRegressor(), #Corrected position of params
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        'Support Vector Regression',

        {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
        SVR(), #Params for SVR
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        'XGB Regressor',

        {'n_estimators': 100, 'learning_rate': 0.1},
        XGBRegressor(), #Params for XGBoost
        (X_train, y_train),
        (X_test, y_test)
    )
]

reports = []

for model_name, params, model, train_set, test_set in models:
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    #apply hyperparameters and train the model
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test, y_pred)

    #store the results
    reports.append((model_name, rmse, mae, r2))

X_train.isna().sum()

reports[i][1]
params

mlflow.set_experiment('Car second experiment') #set this name in mlflow website
mlflow.set_tracking_uri('http://127.0.0.1:5000/') #local host uri

for i, element in enumerate(models):
    model_name = element[0]
    params = element[1]
    model = element[2]
    report = reports[i]

    with mlflow.start_run(run_name = model_name):
        mlflow.log_params(params)
        mlflow.log_metrics({'RMSE': report[1],
                            'MAE': report[2],
                            'R2 score': report[3]})
        
        if 'XGB' in model_name:
            mlflow.xgboost.log_model(model, 'model')
        else:
            mlflow.sklearn.log_model(model, 'model')


#Model registration

model_name = 'XGB 100'
run_id = input('Please type RunID')
model_uri = f'runs:/{run_id}/model'

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri=model_uri,name = model_name)


#Load and Test the model

model_name = 'XGB 100'
model_version = 1
model_uri = f'models:/{model_name}/{model_version}'

loaded_model = mlflow.xgboost.load_model(model_uri)
y_pred = loaded_model.predict(X_test)
y_pred[:4]

#Transition the model to production
#here challenger is the alliance given in mflow website

current_model_uri = f'models:/{model_name}@challenger'
production_model_name = 'RF_car'

client = mflow.MLflowClient()
client.copy_model_version(src_model_uri = current_model_uri, dst_name = production_model_name)

model_version = 1
prod_model_uri = f'models:/{production_model_name}@challenger'

loaded_model = mflow.pyfunc.load_model(prod_model_uri)
y_pred = loaded_model.predict(test_input)
y_pred

#Model Deployment 
#create account in dagshub and enter credentials here

import dagshub
dagshub.init(repo_owner = 'Sree',repo_name='mlflow_demo',mlflow=True)

#repeat model fitting steps for dagshub specifically

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

models = [
    (
        'Gradient Boosting Regressor',

        {'n_estimators': 150, 'learning_rate': 0.1},
        GradientBoostingRegressor(), #example of hyperparameters
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        'Random Forest',

        {'n_estimators': 150, 'max_depth': 5},
        RandomForestRegressor(), #Corrected position of params
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        'Support Vector Regression',

        {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
        SVR(), #Params for SVR
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        'XGB Regressor',

        {'n_estimators': 100, 'learning_rate': 0.1},
        XGBRegressor(), #Params for XGBoost
        (X_train, y_train),
        (X_test, y_test)
    )
]

X_train.isna().sum()
X_train = X_train.dropna()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

for model_name, params, model, train_set, test_set in models:
    try:
        X_train = train_set[0]
        y_train = train_set[1]
        X_test = test_set[0]
        y_test = test_set[1]

        print(f'\nTraining {model_name}...')
        print(f'Parameters: {params}')
        print(f'X_train_shape: {X_train.shape}, y_train_shape: {y_train.shape}')
        print(f'X_test_shape: {X_test.shape}, y_test_shape: {y_test.shape}')
        
        #apply hyperparameters and train the model
        model.set_params(**params) #Set model parameters
        model.fit(X_train, y_train) #train model
        y_pred = model.predict(X_test) #make predictions

        #calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        mae = mean_absolute_error(y_test,y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'RMSE: {rmse}, MAE: {mae}, R2: {r2}')
        #store the results
        reports.append((model_name, rmse, mae, r2))
    except Exception as e:
        print(f'Error with model {model_name}: {e}') #log the error


X_train.isna().sum()



