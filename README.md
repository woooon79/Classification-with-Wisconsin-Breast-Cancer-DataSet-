# Classification-with-Wisconsin-Breast-Cancer-DataSet (PHW1)

## Introduction 
Breast cancer is the second leading cause of cancer death among women. And the incidence of breast cancer continues to increase every year. Therefore, improving diagnostic accuracy by analyzing breast cancer data is an important task. This analysis aims to find best model which classifies whether the breast cancer is benign or malignant. I applied various scalers, hyperparameters, and classifiers to derive the best combination model with the highest accuracy.

## Identify the Dataset
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

###  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)

- Number of Attributes: 10 and class attribute
Attributes 2 through 10 have been used to represent instances. Each instance has one of 2 possible classes: benign or malignant.

- Number of Instances: 699 
- Missing attribute values: 16
    There are 16 instances in Groups 1 to 6 that contain a single missing 
    (i.e., unavailable) attribute value, now denoted by "?".

## Before run this manual, please make sure the install and import following packages.
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing



## Clean and prepare a dataset 
I defined ‘cleaning’ function to modify the dataset. You can add columns names on the dataframe and remove needless features. Also you can handle missing values with it. In this process I just drop the rows of missing values because dataset has only 16 of missing values. After function cleans a dataset, it returns modified dataframe. 

## Training and Testing 
I defined functions to find best model automatically with computing all combination of parameters that specified scaler, encoder, classifiers and hyperparameters. It finds the model that scores the best. In this process, I declared three functions: find_best_comb, tuner and find_best_model. 

First, you should call the find_best_comb. This function allows you to designate various scalers, encoders, hyperparameters, and models you want to try. And it defines 'config' dictionary to gather all these stuffs. Then call tuner function to experiment with various combinations.

 tuner function conducts data preprocessing with various encoders and scalers by nested for-loop. When encoder is ‘NON’ value, it does not perform encoding. In loop, it calls ‘find_best_model’ function to find best classifier and hyperparameters with each scaled and encoded dataset.
 
 find_best_model proceeds training and evaluation process. So first, it splits a dataset into training-set and test-set (0.75:0.25). Then it starts training and testing to find best combination of models and hyperparameters. In this process I experienced with DecisionTree classifier(gini,entropy), SVM and logistic regression. And I applied different hyperparameters by grid search and k-fold cross validation. In k-fold cross validation, also I tried diverse values of k.
 
 ## Result
 Logistic regression, SVM and DecisionTreeClassifier(entropy) showed the best performance. Among them, logistic regression was the most common, and through this, it can be seen that logistic regression is less affected by other factors and can produce good performance.
 
