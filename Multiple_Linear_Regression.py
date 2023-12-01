import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

def model_fit_linear():
    dataset = pd.read_csv('./Data/Data.csv')
    dataset.isna().sum()
    #dataset.info()
    X = dataset.drop('ISOaccuracy', axis=1)
    d = {'X': X}
    print(d)
    y = dataset['ISOaccuracy']
    d = {'Y': y}
    print(d)

    categorical_features_X = ["acttype","singleprec"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                    one_hot,
                                    categorical_features_X)],
                                    remainder="passthrough")

    transformed_X = transformer.fit_transform(X)
    d = {'X': pd.DataFrame(transformed_X).head()}
    #print(pd.DataFrame(transformed_X).head())

    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size = 0.25, random_state = 2)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    #print(X_train)
    #print(pd.DataFrame(X_test).head())
    score=regressor.score(X_test,y_test)
    d = {'score': score}
    print(d)
    d = {'regressor.coef_': regressor.coef_, 'regressor.intercept_': regressor.intercept_}
    print(d)
    y_pred = regressor.predict(X_test)
    d = {'y_pred': y_pred, 'y_test': y_test}
    print(pd.DataFrame(d))
