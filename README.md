# Predicting Customer App Subscription
Machine learning model to predict whether a customer would subscribe for app premium features using their App behaviour analysis

**_Problem Statement_** : A company investing in app development can gain a lot of profit if they identify the customers who are 
						  most likely to purchase their premium features. This can be done by using the data of a customer app usage
						  and various other details of the customer like their age.

**_Dataset_** : The dataset (appdata10.csv) contains 50000 rows and 12 columns.
				The values of columns are the data obtained by the company. This data contains details like the usage of app, age of customer,
				whether they have used premium features, did the play minigame etc.
				An another file top_screens.csv contains the names of top screens that is being used in the app. 

**EDA** : Exploratory Data Analysis and feature engineering techniques has been applied on the dataset. The result of it has been exported to new dataset named new_appdata10.csv
				
**_Algorithm_** : The algorithm used in this dataset is Regularized Logistic Regression (lasso).
				  For model improvement the features was standardized in data preprocessing section. k-fold cross validation technique was used to check skill of the prepared model.

**_Visualisation_** : The library used for visualizing the data, confusion matrix etc. is seaborn.

This code is prepared using Spyder.