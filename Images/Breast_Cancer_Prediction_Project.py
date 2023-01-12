# -*- coding: utf-8 -*-
"""Breast_Cancer_Prediction_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12PTFKBdk6Xrb9jmkE2o6hjnh5IYhUjh_

# Breast Cancer Detection with Machine Learning Algorithms
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization

"""# Load Dataset"""

df = pd.read_csv("/content/Cancer_Data")

df.head()

df.shape

df.tail()

#Index names of the dataset
df.columns

"""# Dataset Summary"""

df = df.drop('Unnamed: 0', axis=1)

df.info()

df['Unnamed: 32']

df = df.drop('Unnamed: 32', axis=1)

df.columns

df = df.drop('id', axis=1)
         #or
#df.drop('id', axis = 1, inplace = True)

df.columns

type(df.columns)

l = list(df.columns)
print(l)

features_mean = l[1:11]

features_se = l[11:20]

features_worst = l[21:]

print(features_mean)

print(features_se)

print(features_worst)

df.head(2)

"""# Relationship between the different features and the diagnosis of the tumor"""

# Relationship between the different features and the diagnosis of the tumor

# Create a scatter plot matrix
sns.pairplot(df, hue='diagnosis', vars=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])

# Show the plot
plt.show()

"""# Bar plot to analyze the data and compare the distribution of malignant & benign tumors """

# Bar plot to analyze the data and compare the distribution of malignant and benign tumors 
# by features radius_mean, area_mean, concavity_mean.
# Create separate dataframes for malignant and benign tumors
malignant = df[df['diagnosis'] == 'M']
benign = df[df['diagnosis'] == 'B']

# Select the features of interest (e.g. radius_mean, area_mean, concavity_mean)
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

# Create a figure with subplots
fig, axes = plt.subplots(2, 5, figsize=(18, 10))

# Iterate through the features
for i, feature in enumerate(features[:int(len(features)/2)]):
    # Create a bar chart for each feature
    ax = axes[0, i]
    ax.bar(['Malignant', 'Benign'], [malignant[feature].mean(), benign[feature].mean()], yerr=[malignant[feature].sem(), benign[feature].sem()], color=['#3574a0', '#e1812d'])
    ax.set_xlabel('Diagnosis')
    ax.set_title(feature)
    
for i, feature in enumerate(features[int(len(features)/2):]):
    # Create a bar chart for each feature
    ax = axes[1, i]
    ax.bar(['Malignant', 'Benign'], [malignant[feature].mean(), benign[feature].mean()], yerr=[malignant[feature].sem(), benign[feature].sem()], color=['#3574a0', '#e1812d'])
    ax.set_xlabel('Diagnosis')
    ax.set_title(feature)


# Show the figure
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x='radius_mean',y='area_mean',data=df,hue='diagnosis',ax=ax)
plt.xlabel('radius_mean')
plt.ylabel('area_mean')
plt.title('Scatter plot: radius_mean vs area_mean')
plt.show()

df['diagnosis'].unique()
# M = Malignant, B = Benign

figsize = (4, 8)
fig, ax = plt.subplots(figsize=figsize)
sns.countplot(df['diagnosis'],label='Count');

df['diagnosis'].value_counts()

df.shape

"""# Explore the data"""

# Summary of all the columns
df.describe()

len(df.columns)

# Correlation Plot

corr = df.corr()
corr
print(corr)

corr.shape

plt.figure(figsize=(8,8))
sns.heatmap(corr);

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

df.head()

df['diagnosis'].unique()

X = df.drop('diagnosis',axis = 1)
X.head()

y = df['diagnosis']
y.head()

"""# Split dataset into training and test set"""

# # split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

df.shape

X_train.shape

X_test.shape

y_train.shape

y_test.shape

X_train.head(1)

from sklearn.preprocessing import StandardScaler
ss =StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

X_train

"""# Machine learning models

# 1. Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

y_pred

y_test

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

lr_acc = accuracy_score(y_test,y_pred)
print(lr_acc)

results = pd.DataFrame()
results

tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'],'Accuracy':[lr_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm','Accuracy']]
results

"""# Decision Tree Classifier

"""

from sklearn.tree import DecisionTreeClassifier
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)

tempResults = pd.DataFrame({'Algorithm':['Decision Tree Classifier Method'],'Accuracy':[lr_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm','Accuracy']]
results

"""# Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)

from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)

tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results

"""# Support Vector Classifier"""

from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)

tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results

"""# Comparison of Algorithm Accuracies"""

import matplotlib.pyplot as plt

# List of algorithms
algorithms = ["Logistic Regression", "Decision Tree", "SVM", "Random Forest"]

# List of accuracy scores
accuracies = [0.982456, 0.982456, 0.982456, 0.959064]

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Create a bar chart to visualize the results
colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
for i, algo in enumerate(algorithms):
    ax1.bar(algo, accuracies[i], color=colors[i])


# Add labels and title to the bar chart
ax1.set_xlabel("Algorithm")
ax1.set_ylabel("Accuracy")
ax1.set_title("Comparison of Algorithm Accuracies (Bar Chart)")

# Create a line plot to visualize the results
ax2.plot(algorithms, accuracies, color='#ff0000')

# Add labels and title to the line plot
ax2.set_xlabel("Algorithm")
ax2.set_ylabel("Accuracy")
ax2.set_title("Comparison of Algorithm Accuracies (Line Plot)")

# Show the figure
plt.show()

