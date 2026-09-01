####################################################################################################################################################################
# Created by Nikhil Myadam
####################################################################################################################################################################

# File for generating the correlation matrix, and surrogate model training and development

####################################################################################################################################################################
# Modules and libraries
####################################################################################################################################################################
import os
import pickle
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

from string import ascii_letters
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

font = {'family': 'Arial', 'size': 11}  # text in graphic, font and size
mpl.rc('font', **font)

####################################################################################################################################################################
# Data Preprocessing
####################################################################################################################################################################

# splitting data into training and testing for IPCF XGB Model
def f_metamodel_data(df_data):
    X = df_data[['Angle_degrees', 'Thickness']]
    y = df_data[['Peak_Impact_Force_SillAssm_N']]
    # dividing the force values by 1000 to convert units from 'N' to 'kN'
    y = y/1000.0

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train1, X_test1, y_train1, y_test1

# splitting data into training and testing for Energy XGB Model
def e_metamodel_data(df_data):
    X = df_data[['Angle_degrees', 'Thickness']]
    y = df_data[['Energy_Absorbed_J']]

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train2, X_test2, y_train2, y_test2

# splitting data into training and testing for maximum/total deformation (i.e. sill assembly) XGB Model
def td_metamodel_data(df_data):
    X = df_data[['Angle_degrees', 'Thickness']]
    y = df_data[['Maximum_Deformation_m']]

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train3, X_test3, y_train3, y_test3

####################################################################################################################################################################
# XGBoost Models
####################################################################################################################################################################

# XGB Model for studying and predicting Peak impact forces
def f_xgb_regressor(f_train_data, f_test_data):
    parameters = {'booster': ['gbtree'],
                  'objective': ['reg:squarederror'],
                  'eval_metric': ['rmse'],
                  'nthread': [3], # Limit XGB Model to use 3 CPU cores
                  'gamma': [0],
                  'subsample': [0.65],
                  'colsample_bytree': [1],
                  'reg_lambda': [1],
                  'reg_alpha': [0],
                  'n_estimators': [400, 500, 600],
                  'max_depth': [8, 9, 10],
                  'min_child_weight': [1],
                  'learning_rate': [0.1, 0.01, 0.015],
                  'max_leaves': [8, 9, 10],
                  'early_stopping_rounds': [10, 25, 50]} # hyperparameters for the model
    
    # defining the xgboost regression model object
    f_xgr = xgb.XGBRegressor()

    # using GridSearch for hyperparameter tuning
    # Limit the number of multiple experiments that GridSearch runs concurrently to 3 i.e. using n_jobs
    f_reg = GridSearchCV(estimator=f_xgr, param_grid=parameters, cv=3, scoring="neg_mean_absolute_error")
    evaluation = [(f_train_data[0], f_train_data[1]), (f_test_data[0], f_test_data[1])]
    f_reg.fit(f_train_data[0], f_train_data[1], eval_set=evaluation, verbose=1)

    # predictions and performance metrics
    # for the training dataset
    f_train_predictions = f_reg.predict(f_train_data[0])
    # reshape the numpy array for data processing and plotting
    f_train_predictions = f_train_predictions.reshape(-1, 1)
    f_train_acc = r2_score(f_train_data[1], f_train_predictions)

    # for the testing dataset
    f_predictions = f_reg.predict(f_test_data[0])
    # reshape the numpy array for data processing and plotting
    f_predictions = f_predictions.reshape(-1, 1)
    f_acc = r2_score(f_test_data[1], f_predictions)
    f_mae = mean_absolute_error(f_test_data[1], f_predictions)
    f_mse = mean_squared_error(f_test_data[1], f_predictions)
    f_rmse = root_mean_squared_error(f_test_data[1], f_predictions)
    f_mape = mean_absolute_percentage_error(f_test_data[1], f_predictions)
    f_best_estimator = f_reg.best_estimator_

    save_variables_model(y_test_model=f_test_data[1], predictions_model=f_predictions, train_acc=f_train_acc, test_acc=f_acc, mae=f_mae, mse=f_mse, rmse=f_rmse, mape=f_mape,
                         best_estimator=f_best_estimator, model_name=f_xgb_regressor.__name__)
    
    save_model(f_best_estimator, model_name=f_xgb_regressor.__name__)

    print('Size of f_test_data[1] is: ', f_test_data[1].shape)
    print('Dtype of f_test_data[1] is: ', type(f_test_data[1]))
    print('Size of f_predictions is: ', f_predictions.shape)
    print('Dtype of f_predictions is: ', type(f_predictions))

    # model validation and plotting results
    plot_data_force(y_test_model=f_test_data[1], predictions_model=f_predictions, model_name=f_xgb_regressor.__name__)
    
    plot_residual_data_force(y_test_model=f_test_data[1], predictions_model=f_predictions, model_name=f_xgb_regressor.__name__)

    force_residuals_summary(y_test_model=f_test_data[1], predictions_model=f_predictions, model_name=f_xgb_regressor.__name__)

# XGB Model for studying and predicting Energy absorption
def e_xgb_regressor(e_train_data, e_test_data):
    parameters = {'booster': ['gbtree'],
                  'objective': ['reg:squarederror'],
                  'eval_metric': ['rmse'],
                  'nthread': [3], # Limit XGB Model to use 3 CPU cores
                  'gamma': [0],
                  'subsample': [0.65],
                  'colsample_bytree': [1],
                  'reg_lambda': [1],
                  'reg_alpha': [0],
                  'n_estimators': [400, 500, 600],
                  'max_depth': [8, 9, 10],
                  'min_child_weight': [1],
                  'learning_rate': [0.1, 0.01, 0.015],
                  'max_leaves': [8, 9, 10],
                  'early_stopping_rounds': [10, 25, 50]} # hyperparameters for the model
    
    # defining the xgboost regression model object
    e_xgr = xgb.XGBRegressor()

    # using GridSearch for hyperparameter tuning
    # Limit the number of multiple experiments that GridSearch runs concurrently to 3 i.e. using n_jobs
    e_reg = GridSearchCV(estimator=e_xgr, param_grid=parameters, cv=3, scoring="neg_mean_absolute_error")
    evaluation = [(e_train_data[0], e_train_data[1]), (e_test_data[0], e_test_data[1])]
    e_reg.fit(e_train_data[0], e_train_data[1], eval_set=evaluation, verbose=1)

    # predictions and performance metrics
    # for the training dataset
    e_train_predictions = e_reg.predict(e_train_data[0])
    # reshape the numpy array for data processing and plotting
    e_train_predictions = e_train_predictions.reshape(-1, 1)
    e_train_acc = r2_score(e_train_data[1], e_train_predictions)

    # for the testing dataset
    e_predictions = e_reg.predict(e_test_data[0])
    # reshape the numpy array for data processing and plotting
    e_predictions = e_predictions.reshape(-1, 1)
    e_acc = r2_score(e_test_data[1], e_predictions)
    e_mae = mean_absolute_error(e_test_data[1], e_predictions)
    e_mse = mean_squared_error(e_test_data[1], e_predictions)
    e_rmse = root_mean_squared_error(e_test_data[1], e_predictions)
    e_mape = mean_absolute_percentage_error(e_test_data[1], e_predictions)
    e_best_estimator = e_reg.best_estimator_

    save_variables_model(y_test_model=e_test_data[1], predictions_model=e_predictions, train_acc=e_train_acc, test_acc=e_acc, mae=e_mae, mse=e_mse, rmse=e_rmse, mape=e_mape,
                         best_estimator=e_best_estimator, model_name=e_xgb_regressor.__name__)
    
    save_model(e_best_estimator, model_name=e_xgb_regressor.__name__)

    print('Size of e_test_data[1] is: ', e_test_data[1].shape)
    print('Dtype of e_test_data[1] is: ', type(e_test_data[1]))
    print('Size of e_predictions is: ', e_predictions.shape)
    print('Dtype of e_predictions is: ', type(e_predictions))

    # model validation and plotting results
    plot_data_energy(y_test_model=e_test_data[1], predictions_model=e_predictions, model_name=e_xgb_regressor.__name__)

    plot_residual_data_energy(y_test_model=e_test_data[1], predictions_model=e_predictions, model_name=e_xgb_regressor.__name__)

    energy_residuals_summary(y_test_model=e_test_data[1], predictions_model=e_predictions, model_name=e_xgb_regressor.__name__)

# XGB Model for studying and predicting Total deformation
def td_xgb_regressor(td_train_data, td_test_data):
    parameters = {'booster': ['gbtree'],
                  'objective': ['reg:squarederror'],
                  'eval_metric': ['rmse'],
                  'nthread': [3], # Limit XGB Model to use 3 CPU cores
                  'gamma': [0],
                  'subsample': [0.65],
                  'colsample_bytree': [1],
                  'reg_lambda': [1],
                  'reg_alpha': [0],
                  'n_estimators': [400, 500, 600],
                  'max_depth': [6, 7, 8],
                  'min_child_weight': [1],
                  'learning_rate': [0.1, 0.01, 0.015],
                  'max_leaves': [8, 9, 10],
                  'early_stopping_rounds': [10, 25, 50]} # hyperparameters for the model
    
    # defining the xgboost regression model object
    td_xgr = xgb.XGBRegressor()

    # using GridSearch for hyperparameter tuning
    # Limit the number of multiple experiments that GridSearch runs concurrently to 3 i.e. using n_jobs
    td_reg = GridSearchCV(estimator=td_xgr, param_grid=parameters, cv=3, scoring="neg_mean_absolute_error")
    evaluation = [(td_train_data[0], td_train_data[1]), (td_test_data[0], td_test_data[1])]
    td_reg.fit(td_train_data[0], td_train_data[1], eval_set=evaluation, verbose=1)

    # predictions and performance metrics
    # for the training dataset
    td_train_predictions = td_reg.predict(td_train_data[0])
    # reshape the numpy array for data processing and plotting
    td_train_predictions = td_train_predictions.reshape(-1, 1)
    td_train_acc = r2_score(td_train_data[1], td_train_predictions)

    # for the testing dataset
    td_predictions = td_reg.predict(td_test_data[0])
    # reshape the numpy array for data processing and plotting
    td_predictions = td_predictions.reshape(-1, 1)
    td_acc = r2_score(td_test_data[1], td_predictions)
    td_mae = mean_absolute_error(td_test_data[1], td_predictions)
    td_mse = mean_squared_error(td_test_data[1], td_predictions)
    td_rmse = root_mean_squared_error(td_test_data[1], td_predictions)
    td_mape = mean_absolute_percentage_error(td_test_data[1], td_predictions)
    td_best_estimator = td_reg.best_estimator_

    save_variables_model(y_test_model=td_test_data[1], predictions_model=td_predictions, train_acc=td_train_acc, test_acc=td_acc, mae=td_mae, mse=td_mse, rmse=td_rmse, mape=td_mape,
                         best_estimator=td_best_estimator, model_name=td_xgb_regressor.__name__)
    
    save_model(td_best_estimator, model_name=td_xgb_regressor.__name__)

    print('Size of td_test_data[1] is: ', td_test_data[1].shape)
    print('Dtype of td_test_data[1] is: ', type(td_test_data[1]))
    print('Size of td_predictions is: ', td_predictions.shape)
    print('Dtype of td_predictions is: ', type(td_predictions))

    # model validation and plotting results
    plot_data_t_deformation(y_test_model=td_test_data[1], predictions_model=td_predictions, model_name=td_xgb_regressor.__name__)

####################################################################################################################################################################
# Model Validation & Results
####################################################################################################################################################################

def plot_data_force(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(15*cm, 15*cm))

    # initializing arrays to store and plot data
    # changing the datatype of y_test_model to numpy array as it is a pandas dataframe
    y_test_model = y_test_model.to_numpy()

    # checking the type and size of all arrays
    print("Shape of y_test_model for IPCF is: ", y_test_model.shape)
    print("Shape of predictions_model for IPCF is: ", predictions_model.shape)
    print('The dtype of y_test_model for IPCF is: ', type(y_test_model))
    print('The dtype of predictions_model for IPCF is: ', type(predictions_model))
  
    # checking true to predicted values of peak impact forces by plotting a diagonal line of slope +45 degrees, representing perfect fit
    plt.scatter(y_test_model, predictions_model, color = 'black', linewidth = 2, label = 'IPCF Data')
    a = np.linspace(min(y_test_model), max(y_test_model), 100)
    plt.plot(a, a, color = 'red', linewidth = 2, linestyle = '--', label = 'Perfect fit line')
    plt.legend(loc='upper left')
    plt.xlabel('True IPCF Values')
    plt.ylabel('Predicted IPCF Values')
    plt.title('Testing the IPCF XGB model')
    plt.tight_layout()
    filename = str(model_name) + '_IPCF_predictions_vs_true.png'
    plt.savefig(path + '/' + filename, dpi=150)
    plt.show()

def plot_data_energy(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(15*cm, 15*cm))

    # initializing arrays to store data
    # changing the datatype of y_test_model to numpy array as it is a pandas dataframe
    y_test_model = y_test_model.to_numpy()

    # checking the size of all arrays
    print("Shape of y_test_model for Energy is: ", y_test_model.shape)
    print("Shape of predictions_model for Energy is: ", predictions_model.shape)
    print('The dtype of y_test_model for Energy is: ', type(y_test_model))
    print('The dtype of predictions_model for Energy is: ', type(predictions_model))    

    # checking true to predicted values of energy by plotting a diagonal line of slope +45 degrees, representing perfect fit
    plt.scatter(y_test_model, predictions_model, color = 'black', linewidth = 2, label = 'Energy Data')
    b = np.linspace(min(y_test_model), max(y_test_model), 100)
    plt.plot(b, b, color = 'red', linewidth = 2, linestyle = '--', label = 'Perfect fit line')
    plt.legend(loc='upper left')
    plt.xlabel('True Energy Values')
    plt.ylabel('Predicted Energy Values')
    plt.title('Testing the Energy XGB model')
    plt.tight_layout()
    filename = str(model_name) + '_Energy_predictions_vs_true.png'
    plt.savefig(path + '/' + filename, dpi=150)
    plt.show()

def plot_data_t_deformation(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(15*cm, 15*cm))

    # initializing arrays to store data
    # changing the datatype of y_test_model to numpy array as it is a pandas dataframe
    y_test_model = y_test_model.to_numpy()

    # checking the size of all arrays
    print("Shape of y_test_model for Total deformation is: ", y_test_model.shape)
    print("Shape of predictions_model for Total deformation is: ", predictions_model.shape)
    print('The dtype of y_test_model for Total deformation is: ', type(y_test_model))
    print('The dtype of predictions_model for Total deformation is: ', type(predictions_model))    

    # checking true to predicted values of total deformation by plotting a diagonal line of slope +45 degrees, representing perfect fit
    plt.scatter(y_test_model, predictions_model, color = 'black', linewidth = 2, label = 'Total deformation Data')
    c = np.linspace(min(y_test_model), max(y_test_model), 100)
    plt.plot(c, c, color = 'red', linewidth = 2, linestyle = '--', label = 'Perfect fit line')
    plt.legend(loc='upper left')
    plt.xlabel('True Total deformation Values')
    plt.ylabel('Predicted Total deformation Values')
    plt.title('Testing the TD XGB model')
    plt.tight_layout()
    filename = str(model_name) + '_T_deformation_predictions_vs_true.png'
    plt.savefig(path + '/' + filename, dpi=150)
    plt.show()

#####################################################################################################################################################################
# Regression Analysis & Residual Plots
#####################################################################################################################################################################
# Residual of predicted-true values should be a normal distribution 
# testing the model for residuals between the ideal/expected output and predicted output

def plot_residual_data_force(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(15*cm, 15*cm))

    # initializing arrays to store and plot data
    # changing the datatype of y_test_model to numpy array as it is a pandas dataframe
    y_test_model = y_test_model.to_numpy()
    
    # check the type, and true and predicted values of PCF
    print('Shape of y_test_model for force residuals is: ', y_test_model.shape)
    print('The dtype of y_test_model for force residuals is: ', type(y_test_model))
    print('Shape of predictions_model for force residuals is: ', predictions_model.shape)
    print('The dtype of predictions_model for force residuals is: ', type(predictions_model))

    # calculating and plotting the residual data
    residual_data_force = (y_test_model - predictions_model)
    
    # check type, size and values of force residuals
    print('The dtype of residual_data_force is: ', type(residual_data_force))
    print('The shape of residual_data_force is: ', residual_data_force.shape)
    print(residual_data_force)

    # can create custom number of discrete bins or set binsizes using the keywords 'binwidth' and 'bins'
    sns.displot(residual_data_force, legend=False, kde=True)
    plt.legend('IPCF_R', loc='upper left')
    plt.xlabel('IPCF Residuals', font=font)
    plt.yscale('symlog')
    plt.title('Error residual distribution for Peak Crushing Forces', font=font)
    filename = str(model_name) + '_IPCF_Residuals.png'
    plt.savefig(path + '/' + filename, dpi=1000, bbox_inches='tight')
    plt.show()

def force_residuals_summary(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))

    y_test_model = y_test_model.to_numpy()
    residual_data_force = (y_test_model - predictions_model)

    # Calculate summary statistics
    mean_force_residual = np.mean(residual_data_force)
    median_force_residual = np.median(residual_data_force)
    std_dev_force_residual = np.std(residual_data_force)
    min_force_residual = np.min(residual_data_force)
    max_force_residual = np.max(residual_data_force)

    # input text

    mean_force_residual_str = repr(mean_force_residual)
    median_force_residual_str = repr(median_force_residual)
    std_dev_force_residual_str = repr(std_dev_force_residual)
    min_force_residual_str = repr(min_force_residual)
    max_force_residual_str = repr(max_force_residual)

    # open file
    filename = str(model_name) + '_IPCF_residuals_summary.txt'
    file = open(path + '/' + filename, "w")

    # convert variable to string
    file.write('Peak Impact Force Residuals' '\n')
    file.write('************************************************************************************************\n')
    file.write('Mean: ' + str(model_name) + ' = ' + mean_force_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Median: ' + str(model_name) + ' = ' + median_force_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Standard Deviation: ' + str(model_name) + ' = ' + std_dev_force_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Minimum Residual: ' + str(model_name) + ' = ' + min_force_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Maximum Residual: ' + str(model_name) + ' = ' + max_force_residual_str + "\n")
    file.write('************************************************************************************************\n')

    # close file
    file.close()

def plot_residual_data_energy(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(15*cm, 15*cm))

    # initializing arrays to store and plot data
    # changing the datatype of y_test_model to numpy array as it is a pandas dataframe
    y_test_model = y_test_model.to_numpy()

    # check the type, and true and predicted values of Energy
    print('Shape of y_test_model for energy residuals is: ', y_test_model.shape)
    print('The dtype of y_test_model for energy residuals is: ', type(y_test_model))
    print('Shape of predictions_model for energy residuals is: ', predictions_model.shape)
    print('The dtype of predictions_model for energy residuals is: ', type(predictions_model))

    # calculating and plotting the residual data
    residual_data_energy = (y_test_model - predictions_model)

    # check type, size and values of energy residuals
    print('The dtype of residual_data_energy is: ', type(residual_data_energy))
    print('The shape of residual_data_energy is: ', residual_data_energy.shape)
    print(residual_data_energy)
    
    # can create custom number of discrete bins or set binsizes using the keywords 'binwidth' and 'bins'
    sns.displot(residual_data_energy, legend=False, kde=True)
    plt.legend('Energy_R', loc='upper left')
    plt.xlabel('Energy Residuals', font=font)
    plt.yscale('symlog')
    plt.title('Error residual distribution for Energy', font=font)
    filename = str(model_name) + '_Energy_Residuals.png'
    plt.savefig(path + '/' + filename, dpi=1000, bbox_inches='tight')
    plt.show()

def energy_residuals_summary(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))

    y_test_model = y_test_model.to_numpy()
    residual_data_energy = (y_test_model - predictions_model)

    # Calculate summary statistics
    mean_energy_residual = np.mean(residual_data_energy)
    median_energy_residual = np.median(residual_data_energy)
    std_dev_energy_residual = np.std(residual_data_energy)
    min_energy_residual = np.min(residual_data_energy)
    max_energy_residual = np.max(residual_data_energy)

    # input text

    mean_energy_residual_str = repr(mean_energy_residual)
    median_energy_residual_str = repr(median_energy_residual)
    std_dev_energy_residual_str = repr(std_dev_energy_residual)
    min_energy_residual_str = repr(min_energy_residual)
    max_energy_residual_str = repr(max_energy_residual)

    # open file
    filename = str(model_name) + '_Energy_residuals_summary.txt'
    file = open(path + '/' + filename, "w")

    # convert variable to string
    file.write('Energy Absorption Residuals' '\n')
    file.write('************************************************************************************************\n')
    file.write('Mean: ' + str(model_name) + ' = ' + mean_energy_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Median: ' + str(model_name) + ' = ' + median_energy_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Standard Deviation: ' + str(model_name) + ' = ' + std_dev_energy_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Minimum Residual: ' + str(model_name) + ' = ' + min_energy_residual_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Maximum Residual: ' + str(model_name) + ' = ' + max_energy_residual_str + "\n")
    file.write('************************************************************************************************\n')

    # close file
    file.close() 

####################################################################################################################################################################
# Correlation Matrix
####################################################################################################################################################################

def plot_correlation_matrix(corr_data):
    path = os.path.join(os.getcwd(), 'Results')

    if not os.path.exists(path):
        os.makedirs(path)

    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(16*cm, 16*cm))
    
    # set the theme
    sns.set_theme(style="whitegrid")
    # Compute the correlation matrix
    corr = corr_data.corr()
    # Mask the lower triangle to display the upper-triangular correlation matrix
    mask = np.tril(np.ones_like(corr, dtype=bool))
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, cmap='Blues', vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt=".1f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    filename = 'Correlation_Matrix.png'
    plt.savefig(os.path.join(path, filename), dpi=1000, bbox_inches='tight')
    plt.show()
    
####################################################################################################################################################################
# XGBoost Model and Data
####################################################################################################################################################################

def save_model(model, model_name):

    model_save_folder = os.getcwd() + '/Models/' + str(model_name)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    model_path = os.path.join(model_save_folder, str(model_name) + '_model.json')
    model.save_model(model_path)
    print("Saved " + str(model_name) + " model to disk: ", model_path)

def save_variables_model(y_test_model, predictions_model, train_acc, test_acc, mae, mse, rmse, mape, best_estimator, model_name):

    variables_save_folder = os.getcwd() + '/Results/' + str(model_name)
    if not os.path.exists(variables_save_folder):
        os.makedirs(variables_save_folder)
    
    path, _, _ = next(os.walk(variables_save_folder))

    # input text
    y_test_str = repr(y_test_model[:])
    predictions_str = repr(predictions_model[:])
    acc_str1 = repr(train_acc)
    acc_str2 = repr(test_acc)
    mae_str = repr(mae)
    mse_str = repr(mse)
    rmse_str = repr(rmse)
    mape_str = repr(mape)
    est_str = repr(best_estimator)

    # open file
    filename = str(model_name) + '_performance.txt'
    file = open(path + '/' + filename, "w")

    # convert variable to string
    file.write('************************************************************************************************\n')
    file.write('Test Data: ' + str(model_name) + ' = ' + y_test_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Predicted Data: ' + str(model_name) + ' = ' + predictions_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Accuracy of Model_Train data: ' + str(model_name) + ' = ' + acc_str1 + "\n")
    file.write('************************************************************************************************\n')
    file.write('Accuracy of Model_Test data: ' + str(model_name) + ' = ' + acc_str2 + "\n")
    file.write('************************************************************************************************\n')
    file.write('MAE of Model: ' + str(model_name) + ' = ' + mae_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('MSE of Model: ' + str(model_name) + ' = ' + mse_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('RMSE of Model: ' + str(model_name) + ' = ' + rmse_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('MAPE of Model: ' + str(model_name) + ' = ' + mape_str + "\n")
    file.write('************************************************************************************************\n')
    file.write('Best estimator of Model: ' + str(model_name) + ' = ' + est_str + "\n")
    file.write('************************************************************************************************\n')

    # close file
    file.close()

#####################################################################################################################################################################
# Performance Metrics
#####################################################################################################################################################################

def mean_absolute_error(y_test_model, predictions_model):
    mae = np.mean(abs(y_test_model - predictions_model))
    return mae

def mean_squared_error(y_test_model, predictions_model):
    mse = np.square(np.subtract(y_test_model, predictions_model)).mean()
    return mse

def root_mean_squared_error(y_test_model, predictions_model):
    rmse = np.sqrt(np.mean(np.square(y_test_model - predictions_model)))
    return rmse

def mean_absolute_percentage_error(y_test_model, predictions_model):
    mape = np.mean(np.abs((y_test_model - predictions_model) / y_test_model)) * 100
    return mape

#####################################################################################################################################################################

if __name__ == '__main__':

    # Note: Replace <PATH_TO_PROJECT> with the appropriate local project path.
    
    df = pd.read_csv(r'<PATH_TO_PROJECT>\XGB_MODELS_DATABASE.csv')
    
    # Drop the index column if it exists
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Reset index after dropping unnecessary columns
    df.dropna(inplace=True)
    # df.reset_index(drop=True, inplace=True)
    print(df.keys())
    
    # Initializing the training and testing data for all xgb models 
    X_train_f, X_test_f, y_train_f, y_test_f = f_metamodel_data(df_data = df)
    X_train_e, X_test_e, y_train_e, y_test_e = e_metamodel_data(df_data = df)
    X_train_td, X_test_td, y_train_td, y_test_td = td_metamodel_data(df_data = df)

    # Training and testing data to study and predict IPCF
    f_train_data_in = [X_train_f, y_train_f]
    f_test_data_in = [X_test_f, y_test_f]

    # Training and testing data to study and predict Energy Absorption
    e_train_data_in = [X_train_e, y_train_e]
    e_test_data_in = [X_test_e, y_test_e]

    # Training and testing data to study and predict Total deformation
    td_train_data_in = [X_train_td, y_train_td]
    td_test_data_in = [X_test_td, y_test_td]

    # Initializing the XGB models
    f_xgb_regressor(f_train_data=f_train_data_in, f_test_data=f_test_data_in)
    e_xgb_regressor(e_train_data=e_train_data_in, e_test_data=e_test_data_in)
    td_xgb_regressor(td_train_data=td_train_data_in, td_test_data=td_test_data_in)
    
    # plotting correlation matrix
    plot_correlation_matrix(corr_data=df)