# First we have to have a deep understanding of the problem. This problem is a binary classification problem with real-life dirty data given to us as train and test. A dictionary of the features and their collection practices have not been provided. Our first objective would be to perform Exploratory Data Analysis (EDA) to clean the data and make it suitable before feeding it to any ML model.

# Importing python libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
%matplotlib inline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

# Loading the train/test datasets
train_data = pd.read_csv('exercise_40_train.csv')
test_data = pd.read_csv('exercise_40_test.csv')
pd.set_option('display.max_columns', None)
pd.options.display.max_rows = 1000

# Removing columns with these consitions: 1. low variance 2. single value 3. more than 80% missing values
string_cols = train_data.select_dtypes(include='O')
unusable = []

# Single value columns
for column in string_cols.columns:
    if len(string_cols[column].value_counts()) == 1:
        unusable.append(column)
        
# Low variance columns
low_variance = ['x58', 'x67', 'x71', 'x84']
unusable.extend(low_variance)
train_data = train_data.drop(unusable, axis=1)

# Missing values more than 80%
miss_thres = 0.2 * len(train_data)
train_data = train_data.dropna(thresh=miss_thres,axis=1)

# Removing the outliers
train_data = train_data[(train_data['x21'] - train_data['x21'].mean())/train_data['x21'].std() <= 3]

# We have both categorical features and numerical features that need to be separated. Next, we fix the values in the "week days" column to be consistent, and we remove the "%" and "$" signs from the 'x7' and 'x19' columns.
cat_cols = train_data.select_dtypes(include='O')
num_cols = train_data.select_dtypes(exclude='O')
pd.DataFrame(cat_cols['x33'].value_counts()).plot(kind='bar', figsize=(10,3))
week = {'Mon': 'Monday', 'Tue': 'Tuesday', 'Wed': 'Wednesday', 'Thur': 'Thursday', 'Fri': 'Friday', 'Sat': 'Saturday', 'Sun': 'Sunday'}
mapping_dic = {'x3': week}
cat_cols = cat_cols.replace(mapping_dic)
cat_cols['x7'] = cat_cols['x7'].str.rstrip('%').astype('float')
cat_cols['x19'] = cat_cols['x19'].str.lstrip('$').astype('float')
pd.DataFrame(cat_cols['x3'].value_counts()).plot(kind='bar', figsize=(10,3))

# Next, we add the fixed columns to the numerical features dataframe since they include numerical values originally.
num_cols['x7'] = cat_cols['x7']
num_cols['x19'] = cat_cols['x19']
cat_cols.drop(['x7','x19'],axis = 1, inplace = True)

# We can look at all the numerical missing values by plotting them inside a heatmap.
plt.figure(figsize=(30, 15))
sns.heatmap(num_cols.isnull(), cbar=False)

# We perform simple imputation to fill missing numerical values inside the dataframe.
imp = IterativeImputer()
imp_num_df = pd.DataFrame(imp.fit_transform(num_cols))
imp_num_df.index = num_cols.index
imp_num_df.columns = num_cols.columns

# We can look at all the categorical missing values by plotting them inside a heatmap.
plt.figure(figsize=(30, 15))
sns.heatmap(cat_cols.isnull(), cbar=False)

# We perform simple "most frequent" imputation to fill missing categorical values inside the dataframe.
for col in cat_cols:
    max_freq = cat_cols[col].value_counts().index[0]
    cat_cols[col][pd.isna(cat_cols[col])] = max_freq
	
# We convert the categorical features into numerical values using the LabelEncoder package in python.
le_dict = {}
label_df = pd.DataFrame()
for col in cat_cols:
    le = LabelEncoder()
    label_df[col] = le.fit_transform(cat_cols[col])
    le_dict[col] = le
	
# We create a dataframe of binary cols representing instance of each category across multipile columns.
enc = OneHotEncoder()
onehot_train = pd.DataFrame(enc.fit_transform(label_df).toarray())

# We scale our numerical data and we concatenate both categorical and numerical datasets back together.
scalar = StandardScaler()
scaled_num_train = pd.DataFrame(scalar.fit_transform(imp_num_df.iloc[:, 1:]))
x_tr = pd.concat([scaled_num_train, onehot_train], axis = 1)
y_tr = train_data['y'].astype('int')

# We perform the same operation with the test dataset.
cat_cols_test = test_data.select_dtypes(include='O')
num_cols_test = test_data.select_dtypes(exclude='O')
cat_cols_test = cat_cols_test.replace(mapping_dic)
cat_cols_test['x7'] = cat_cols_test['x7'].str.rstrip('%').astype('float')
cat_cols_test['x19'] = cat_cols_test['x19'].str.lstrip('$').astype('float')
num_cols_test['x7'] = cat_cols_test['x7']
num_cols_test['x19'] = cat_cols_test['x19']
cat_cols_test.drop(['x7','x19'],axis = 1, inplace = True)
imp_num_df_test = pd.DataFrame(imp.fit_transform(num_cols_test))
imp_num_df_test.index = num_cols_test.index
imp_num_df_test.columns = num_cols_test.columns
for col in cat_cols_test:
    max_freq = cat_cols_test[col].value_counts().index[0]
    cat_cols_test[col][pd.isna(cat_cols_test[col])] = max_freq
label_df_test = pd.DataFrame()
for col in cat_cols_test:
    le = le_dict[col]
    label_df_test[col] = le.transform(cat_cols_test[col])
onehot_df_test = pd.DataFrame(enc.transform(label_df_test).toarray())
scaled_num_df_test = pd.DataFrame(scalar.transform(imp_num_df_test))
x_te = pd.concat([scaled_num_df_test, onehot_df_test], axis = 1)

# We split out training dataset into train and validation
X_train, X_test, y_train, y_test = train_test_split(x_tr, y_tr, test_size=0.2, random_state=1)

# For the GLM model, we employ recursive feature reduction w/ cross validation.
LR = RFECV(LogisticRegression(max_iter=2000), scoring = 'roc_auc', n_jobs = -1, cv = 3, step = 5)
LR.fit(X_train, y_train)

# We generate the predicted probabilites with the GLM model.
LR_probs = LR.predict_proba(X_test)
print('AUC: ', roc_auc_score(y_test, LR_probs[:,1]))
print('Accuracy: ', LR.score(X_test, y_test))
# Results: AUC=0.811129, Accuracy=0.866625

# For the non-GLM model, we setup a grid search inside the SVM algorithm.
grid_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
SVM = GridSearchCV(SVC(probability=True), grid_parameters, cv=3, scoring='roc_auc', n_jobs = -1)
SVM.fit(X_train, y_train)

# We generate the predicted probabilites with the non-GLM model.
SVM_probs = SVM.predict_proba(X_test)
print('AUC: ', roc_auc_score(y_test, SVM_probs[:,1]))
print('Accuracy: ', SVM.score(X_test, y_test))
# Results: AUC=0.885789, Accuracy=0.885801

# We save the predictions of the GLM and non-GLM models on the test dataset.
LR_probs = LR.predict_proba(x_te)
np.savetxt("glmresults.csv", LR_probs[:,1], delimiter=",")
SVM_probs = SVM.predict_proba(x_te)
np.savetxt("nonglmresults.csv", SVM_probs[:,1], delimiter=",")