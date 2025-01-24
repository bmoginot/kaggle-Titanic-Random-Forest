import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

pass_data = pd.read_csv("train.csv")
pass_data.head()

seed = 23

init_features = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"] # choose relevant features
pass_data = pass_data[init_features]

pass_data = pd.get_dummies(pass_data) # one hot encode categorical values
pass_data.isnull().values.any() # check to see if there are any NaNs
pass_data = pass_data.dropna() # remove rows with NaN values

pass_data.head()

features = ["Pclass", 
            "Age", 
            "SibSp", 
            "Parch", 
            "Sex_female", 
            "Sex_male", 
            "Embarked_C", 
            "Embarked_Q", 
            "Embarked_S"] # update features based on OHE

X = pass_data[features] # values to learn from 
y = pass_data["Survived"] # values to predict

X.shape, y.shape # remember to clean up data before defining variables

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=seed)

# Hyperparameters
n_estimators = 300
max_depth = 18
min_samples_split = 4
min_samples_leaf = 2

rf_model = RandomForestClassifier(n_estimators=n_estimators, 
                                 max_depth=max_depth, 
                                 min_samples_leaf=min_samples_leaf, 
                                 min_samples_split=min_samples_split, 
                                 random_state=seed) # initialize random forest model

rf_model.fit(train_X, train_y) # fit model to training data
rf_predictions = rf_model.predict(val_X) # make predictions based on test data

rf_mae = mean_absolute_error(rf_predictions, val_y) # compare predicted values against ground truth

from csv import writer

with open("log.csv", "a", newline="") as log:
    log_writer = writer(log)
    log_writer.writerow([n_estimators, max_depth, min_samples_split, min_samples_leaf, rf_mae]) # keep track of how parameters influence error

print(rf_mae)
