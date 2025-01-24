import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# define global parameters
n_estimators = 300
max_depth = 18
min_samples_split = 4
min_samples_leaf = 2
seed = 23

def clean_train_data(file):
    data = pd.read_csv(file)
    init_features = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"] # choose relevant features
    data = data[init_features]
    data = pd.get_dummies(data) # one hot encode categorical values
    data = data.dropna() # remove rows with NaN values
    return data

def get_vals(data):
    """get X and y to use in training and testing model"""
    features = ["Pclass", 
                "Age", 
                "SibSp", 
                "Parch", 
                "Sex_female", 
                "Sex_male", 
                "Embarked_C", 
                "Embarked_Q", 
                "Embarked_S"] # update features based on OHE

    X = data[features] # values to learn from 
    y = data["Survived"] # values to predict
    return X, y

def fit_model():
    """fit model to train data then make predictions on test data"""
    rf_model = RandomForestClassifier(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    min_samples_leaf=min_samples_leaf, 
                                    min_samples_split=min_samples_split, 
                                    random_state=seed) # initialize random forest model
    rf_model.fit(X, y) # fit model to training data
    return rf_model

def clean_test_data(file):
    data = pd.read_csv(file)
    init_features = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"] # choose relevant features
    data = data[init_features]
    data = pd.get_dummies(data) # one hot encode categorical values
    data = data.dropna() # remove rows with NaN values
    return data

cleaned_data = clean_train_data("train.csv")
X, y = get_vals(cleaned_data)
model = fit_model()
test_data = clean_test_data("test.csv")
test_X = test_data.drop("PassengerId", axis=1)
preds = model.predict(test_X) # get predictions based on test data

output = pd.DataFrame({'Id': test_data.PassengerId, 'Survived': preds})
output.to_csv('submission.csv', index=False) # write solution out to csv for submission