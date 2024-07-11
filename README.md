# Breast Cancer Prediction with Machine Learning Algorithms

## Introduction
Breast cancer is one of the most common forms of cancer among women worldwide. Early detection and accurate diagnosis of breast cancer are crucial for effective treatment and recovery. In this project, my aim is to develop a machine learning model that can predict the diagnosis of breast cancer using different algorithms. The project is done in Google Colab using libraries such as:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

![Visualizations](https://github.com/PrashanthReddy47/Breast-Cancer-Prediction-Guided-Proejct/blob/main/Images/Final_Results.jpg)

## Table of Contents
- [Introduction](#introduction)
- [Dataset Summary](#dataset-summary)
- [Exploring the Data](#exploring-the-data)
- [Relationship between Features and Diagnosis](#relationship-between-features-and-diagnosis)
- [Splitting the Dataset](#splitting-the-dataset)
- [Machine Learning Models](#machine-learning-models)
- [Comparison of Algorithm Accuracies](#comparison-of-algorithm-accuracies)
- [Conclusion](#conclusion)
- [Data Attribution and Credit](#data-attribution-and-credit)

## Dataset Summary
The [dataset](https://github.com/PrashanthReddy47/Breast_Cancer_Prediction_Project/blob/main/Dataset.csv) contains 569 samples of breast cancer tumors and 30 features. The dataset is divided into two classes: malignant (cancerous) and benign (non-cancerous) tumors. The dataset is loaded into the program using the pandas library.

## Exploring the Data
To explore the data further, we use the seaborn library to create a [heatmap](https://github.com/PrashanthReddy47/Breast-Cancer-Prediction-Guided-Proejct/blob/main/Images/Correlation_Plot.png) that shows the correlation between the different features. The heatmap shows that some features, such as the radius of the tumor, are highly correlated with the diagnosis of the tumor.

## Relationship between Features and Diagnosis
The dataset contains 30 features, including the diagnosis of the tumor [malignant or benign](https://github.com/PrashanthReddy47/Breast-Cancer-Prediction-Guided-Proejct/blob/main/Images/Cancer_Diagnosis_Plot.png). To understand the relationship between the different features and the diagnosis of the tumor, we use a bar plot to analyze the data and compare the distribution of malignant and benign tumors. 
The bar plot shows that some features, such as the radius of the tumor, are more indicative of malignant tumors than benign tumors.

## Splitting the Dataset
To ensure that the model is accurate, we split the dataset into training and test sets. The training set is used to train the model, while the test set is used to evaluate the model's performance.

## Machine Learning Models
In this project, we use four machine learning algorithms to predict the diagnosis of breast cancer: 
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier 
- Support Vector Classifier
  
The results of this project show that all four machine learning algorithms used for predicting the diagnosis of breast cancer, namely Logistic Regression, Decision Tree Classifier, Support Vector Classifier, and Random Forest Classifier, have a similar accuracy of 0.982456. However, when it comes to the Random Forest Classifier Method, its accuracy is slightly lower with 0.959064.

## Comparison of Algorithm Accuracies
To [compare the accuracy](https://github.com/PrashanthReddy47/Breast-Cancer-Prediction-Guided-Proejct/blob/main/Images/Accuracy_Plot.png) of the different algorithms, we use the test set to evaluate the model's performance. The results show that the Logistic Regression has the highest accuracy, followed by the Decision Tree Classifier, the Support Vector Classifier, and the Random Forest Classifier.

## Conclusion
In this project, we developed a machine learning model that can predict the diagnosis of breast cancer using different algorithms. The model was trained on the Wisconsin Diagnosis Breast Cancer (WDBC) dataset and tested on a test set. 
The project demonstrates the potential of machine learning in the early detection and accurate diagnosis of breast cancer. However, it is important to note that the results of this project are based on a specific dataset and may not be generalizable to other populations. 
Overall, this project highlights the importance of utilizing advanced machine learning techniques in the field of medical diagnosis, and the potential for these techniques to improve the quality of care for patients with breast cancer.

## Data Attribution and Credit
This project uses the Wisconsin Diagn Breast Cancer (WDBC) dataset, which was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg. I would like to extend my gratitude to Dr. Wolberg for providing access to this valuable dataset, which has made this research possible.
