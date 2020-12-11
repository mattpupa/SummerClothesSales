#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:13:03 2020

@author: Matt

This project is practice for data science and machine learning.
I wanted to get some practice in data cleansing, exploratory data analysis,
and building a predictive model. The data included in this project is summer
sales data from August 2020.

https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish

In my first project, I created a random forest classifier for property
assessment in Boston. I built a model that predicted whether or not a property
was assessed at more than $750K. It used a combination of a few continuous
features, and many categorical features.

I want to change things up a little now and work with more with continuous
features, and build a regression model. Most likely, this will be a
polynomial regression model or logistic regression model.

Use polynomial to capture interactions between features
Use logistic to get probabilities and do classification

For polynomial, I may try to predict the units sold of the items. If I
decide to, I could also use logistric regression or an SVM to predict whether
the item will sell above a certain count.

"""


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler


scaler = MinMaxScaler()


#import matplotlib.ticker as ticker
#import matplotlib.pyplot as plt
#import datetime as dt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
The first step is importing the csv file with the sales data. There are 
1,573 records and 43 columns.

There is also a csv file that has a row for each unique keyword. The keyword
may play a part in the prediction as well.
"""

# Creates a variable from CSV files and sets any blank values(' ') to NaN
sales = pd.read_csv('/Users/Matt/Desktop/DataScience/Kaggle/SummerClothesSales/sales082020.csv', na_values=' ')
dfsales = pd.DataFrame(sales)

# Creates a variable from CSV files and sets any blank values(' ') to NaN
categoriesbycount = pd.read_csv('/Users/Matt/Desktop/DataScience/Kaggle/SummerClothesSales/categoriesbycount.csv')
dfcategoriesbycount = pd.DataFrame(categoriesbycount)

"""
For the model, I need to get the relevant columns.

"""


df_model = pd.DataFrame(dfsales[['price'
                                 ,'retail_price'
                                 ,'units_sold'
                                 ,'rating'
                                 ,'rating_five_count'
                                 ,'rating_four_count'
                                 ,'rating_three_count'
                                 ,'rating_two_count'
                                 ,'rating_one_count'
                                 ,'badges_count'
                                 ,'countries_shipped_to'
                                 ,'product_variation_inventory'
                                 ,'shipping_option_price'
                                 ,'merchant_rating_count'
                                 ,'merchant_rating'
                                 ]])
df_model.dropna(inplace=True)

y = df_model['units_sold']
X = df_model[['price'
              ,'retail_price'
              ,'rating'
              ,'rating_five_count'
              ,'rating_four_count'
              ,'rating_three_count'
              ,'rating_two_count'
              ,'rating_one_count'
              ,'badges_count'
              ,'countries_shipped_to'
              ,'product_variation_inventory'
              ,'shipping_option_price'
              ,'merchant_rating_count'
              ,'merchant_rating'
              ]]


"""
Will need to play with normalization since this is linear regression
and feature weights are probably different.

Look at Ridge, Lasso, etc.

Will also need to use Scaling since the feature inputs have different
scales...MinMaxScaling. Can this be used on everything or only Ridge
regression?

Merchant rating count is much higher than rating for example, which will
throw the model off

NOTE: regularization is more important with smaller amounts of training data
relative to the number of features in the model. As amount of training data
increases, regularization becomes less critical

use ridge for many features with small/medium sized effects
use lasso for fewer features with medium/large sized effects
higher 'alpha' means more regularization

For logistic regression and linear SVMs, regularization is done with parameter 
'c' higher value of 'c' means less regularization
small c = .1
large c = 100

Kernelized SVM are used when data is messy (on a scatter plot), and it's
difficult to do binary classification on the real world data. Kernelized SVM
can also be used for regression if needed.

Kernelized SVMs transform data to higher dimensional feature space before 
doing the classification. There are different Kernels that result different
transformations. 'rbf' and 'polynomial' are two examples.

'rbf' uses gamma parameter to determine the influence of each training
sample reaches. Small gamma means larger similarity radius, which means
samples further apart are grouped together in a decision boundary.
small gamma = .01
large gamma = 10

"""


scores_list = []

# Create train and test populations
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Linear regression
linreg = LinearRegression().fit(X_train, y_train)
linreg_score_train = linreg.score(X_train, y_train)
linreg_score_test = linreg.score(X_test, y_test)

scores_list.append(['linear regression', np.nan, np.sum(linreg.coef_ != 0)
                    , linreg_score_train, linreg_score_test])



# Ridge regression
linridge = Ridge(alpha=1).fit(X_train, y_train)
linridge_score_train = linridge.score(X_train, y_train)
linridge_score_test = linridge.score(X_test, y_test)


# Ridge regression with normalization

alpha_list = [0, 1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 1000]

# Fit and transform X_train and transform X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=1).fit(X_train_scaled, y_train)
linridge_scaled_score_train = linridge.score(X_train_scaled, y_train)
linridge_scaled_score_test = linridge.score(X_test_scaled, y_test)

def ridge_normalized_by_alpha_scaled():
    for a in alpha_list:
        ridge = Ridge(alpha=a).fit(X_train_scaled, y_train)
        ridge_scaled_score_train = ridge.score(X_train_scaled, y_train)
        ridge_scaled_score_test = ridge.score(X_test_scaled, y_test)
        scores_list.append(['ridge regression', a, np.sum(ridge.coef_ != 0)
                    , ridge_scaled_score_train, ridge_scaled_score_test])
        #print(a, ridge_scaled_score_train, ridge_scaled_score_test)

# Lasso regression
linlasso = Lasso(alpha=1, max_iter = 10000).fit(X_train_scaled, y_train)
linlasso_scaled_score_train = linlasso.score(X_train_scaled, y_train)
linlasso_scaled_score_test = linlasso.score(X_test_scaled, y_test)

def lasso_normalized_by_alpha_scaled():
    for a in alpha_list:
        lasso = Lasso(alpha=a).fit(X_train_scaled, y_train)
        lasso_scaled_score_train = lasso.score(X_train_scaled, y_train)
        lasso_scaled_score_test = lasso.score(X_test_scaled, y_test)
        scores_list.append(['lasso regression', a, np.sum(lasso.coef_ != 0)
                    , lasso_scaled_score_train, lasso_scaled_score_test])
        #print(a, lasso_scaled_score_train, lasso_scaled_score_test)

# Polynomial Regression
# https://en.wikipedia.org/wiki/Degree_of_a_polynomial
degrees = (1,3,6,9)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y,
                                                   random_state = 0)

polyreg = LinearRegression().fit(X_train_poly, y_train_poly)

polyreg_score_train = polyreg.score(X_train_poly, y_train_poly)
polyreg_score_test = polyreg.score(X_test_poly, y_test_poly)

scores_list.append(['polynomial regression', np.nan, np.sum(polyreg.coef_ != 0)
                    , polyreg_score_train, polyreg_score_test])

# Polynomial + Ridge Regression

polyridge = Ridge(alpha=1).fit(X_train_poly, y_train_poly)

polyridge_score_train = polyridge.score(X_train_poly, y_train_poly)
polyridge_score_test = polyridge.score(X_test_poly, y_test_poly)

def polyridge_normalized_by_alpha():
    for a in alpha_list:
        poly_ridge = Ridge(alpha=a).fit(X_train_poly, y_train)
        poly_ridge_scaled_score_train = poly_ridge.score(X_train_poly, y_train)
        poly_ridge_scaled_score_test = poly_ridge.score(X_test_poly, y_test)
        scores_list.append(['poly + ridge', a, np.sum(poly_ridge.coef_ != 0)
                    , poly_ridge_scaled_score_train, poly_ridge_scaled_score_test])
        #print(a, poly_ridge_scaled_score_train, poly_ridge_scaled_score_test)

def polyridge_normalized_by_alpha_scaled():
    for a in alpha_list:
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler.transform(X_test_poly)
        poly_ridge_scaled = Ridge(alpha=a).fit(X_train_poly_scaled, y_train)
        poly_ridge_scaled_score_train = poly_ridge_scaled.score(X_train_poly_scaled, y_train)
        poly_ridge_scaled_score_test = poly_ridge_scaled.score(X_test_poly_scaled, y_test)
        scores_list.append(['poly + ridge scaled', a, np.sum(poly_ridge_scaled.coef_ != 0)
                    , poly_ridge_scaled_score_train, poly_ridge_scaled_score_test])
        #print(a, poly_ridge_scaled_score_train, poly_ridge_scaled_score_test)
        
# Polynomial + Lasso Regression

polylasso = Lasso(alpha=1).fit(X_train_poly, y_train_poly)

polylasso_score_train = polylasso.score(X_train_poly, y_train_poly)
polylasso_score_test = polylasso.score(X_test_poly, y_test_poly)
        
def polylasso_normalized_by_alpha():
    for a in alpha_list:
        poly_lasso = Lasso(alpha=a, max_iter = 10000).fit(X_train_poly, y_train)
        poly_lasso_scaled_score_train = poly_lasso.score(X_train_poly, y_train)
        poly_lasso_scaled_score_test = poly_lasso.score(X_test_poly, y_test)
        scores_list.append(['poly + lasso', a, np.sum(poly_lasso.coef_ != 0)
                    , poly_lasso_scaled_score_train, poly_lasso_scaled_score_test])
        #print(a, poly_lasso_scaled_score_train, poly_lasso_scaled_score_test)
  

ridge_normalized_by_alpha_scaled()
lasso_normalized_by_alpha_scaled()
polyridge_normalized_by_alpha()
polyridge_normalized_by_alpha_scaled()
polylasso_normalized_by_alpha()


df_scores = pd.DataFrame(scores_list, columns=['model_type', 'alpha', 'features_kept'
                                               ,'r2_train', 'r2_test'])


"""
Most of my commentary in this file is notes I took from reviewing the intro 
to machine learning course from the UMich specialization. It was a great 
refresher that reminded me of the nuances between different types of predicted 
models, and how they work. 

For the sales data specifically, I knew I wanted to try to predict the units
sold based on the data I had. Because I wanted to predict a continuous
variable, I needed some type of regression. 

I was happy that I ended up calculating a bunch of different models, and
analyzing their accuracy scores. the data is stored in df_scores. I added
records for each model with the model type, the alpha if applicable, the
number of features kept in the model (the coefficient), the training
accuracy score, and the test accuracy score.

I started off with linear regression, and then added regularization using
ridge and lasso regularization. These models performed as expected.
The linear model had a higher training score accuracy than both the ridge
and lasso regressions regardless of the alpha. However, the linear model
had a LOWER testing score accuracy than all of the ridge and lasso regressions,
regardless of the alpha.

I then moved on to polynomial regression. I don't know exactly if this type
of model is fit for polynomial regression since the features may not have
as clear interaction as in other scenarios. IE. Modeling conversion rates
for a saas company that has product and sales interactions. Regardless,
I thought it would give the model more features to work with.

The polynomial regression had the highest training accuracy score (.93), but
a terrible testing accuracy score (.36), which suggests that the model is
very much overfitting the training data. From there, I added in ridge and
lasso regularization to the polynomial model.

Finally, I added scaling to the models. This was needed since some of the
features are on such different scales. For example, the 'merchant_rating_count'
is much higher than 'badges_count'. 

When comparing all the models, the model with the highest test accuracy score
was the polynomial ridge scaled model (.79 with alpha = 2). In fact, the top
5 scoring models were all polynomial ridge scaled models. This doesn't
surprise me since scaling helps model performance, and ridge regularization
was more relevant. The polynomial model created a lot more feature columns,
and ridge is better to use in that case than lasso.

I do wonder if .79 is an acceptable accuracy score, but I think it's decent
for the data I have. There weren't a ton of features to start with.

The next project is to learn APIs, so when I get more robust data, I'll
investigate and worry about getting a really high testing accuracy score.
For now, I'm happy exploring this data and getting a good refresher.

"""







