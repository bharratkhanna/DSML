# Import specific libraries

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Increase the print output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# Set working Directory

os.chdir("Techstack/ML/property_case/")

# Import dataset 

rawDf = pd.read_csv("PropertyPrice_Data.csv")
predictionDf = pd.read_csv("PropertyPrice_Prediction.csv")
# Analyze Data

rawDf.shape
predictionDf.shape

# Train/Test divide

from sklearn.model_selection import train_test_split

 
# Splitted Randomlly: 
#     - To control randomness, so that your model when broken in test,split 
#     doesn't break on re run with different accuracy, we use random_state
#     - Give any value to random_state it does not effect on the basis low or 
#     high value


trainDf, testDf = train_test_split(rawDf, train_size=0.8, random_state=150)

# Creating one dataset:
    
trainDf["Source"] =  "Train"
testDf["Source"] =  "Train"
predictionDf["Source"] =  "Train"

# Concat all tables:
    
fullRaw = pd.concat([trainDf, testDf, predictionDf], axis=0)
fullRaw.shape

# Identifier Columns: Those columns which doesnt help model to find pattern
# Like: Emp_ID, Account ID, Mobile no, RollNo for identification
 
# Remove Identifier Columns:
fullRaw.drop("Id", axis=1, inplace=True)

# Check for NA's
fullRaw.isnull().sum()

# Check Dtypes
fullRaw.dtypes

################################################
# Univariate Analysis : Missing value imputation
################################################

# Fill NA

# Garage (Categorical Value: Mode)

tempMode = fullRaw.loc[fullRaw["Source"] == "Train","Garage"].mode()[0]
fullRaw["Garage"].fillna(tempMode,inplace=True)

# Automate missing values:
    
for i in fullRaw.columns:
    if i != "Sale_Price":
        if fullRaw[i].dtype != "O":
            tempMedian = fullRaw.loc[fullRaw["Source"] == "Train", i].median()
            fullRaw[i].fillna(tempMedian, inplace=True)
        else:
            tempMode = fullRaw.loc[fullRaw["Source"] ==  "Train", i].mode()[0]
            fullRaw[i].fillna(tempMode, inplace=True)


fullRaw.isna().sum()


#################################################
# Bivariate Analysis: Analysis between 2 columns
################################################# 

# Continuous Variables:

corrDf =  fullRaw[fullRaw["Source"] == "Train"].corr()          

# Heatmap

sns.heatmap(corrDf, 
            xticklabels=(corrDf.columns),
            yticklabels=(corrDf.columns),
            cmap="turbo_r", annot=True) 

plt.show()

 

      