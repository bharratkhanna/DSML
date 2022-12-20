#%%

# Import Libraries
####################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#--------------------------------------------------------------------------------------|
#%%

# Read Dataset
#################

fullRaw = pd.read_csv("BankCreditCard.csv")

#--------------------------------------------------------------------------------------|
# %%

# Sampling - Divide into train/test
#####################################

trainDf,testDf = train_test_split(fullRaw,train_size=0.7, random_state=123)

# Add Source Column
trainDf["Source"] = "Train"
testDf["Source"] = "Test"

# Combine Train,Test
fullRaw = pd.concat([trainDf,testDf],axis = 0)
fullRaw.shape 


#--------------------------------------------------------------------------------------|
# %%

# Univariate Analysis
########################

# Initial Intution
fullRaw.head()  
    # Dependent Column is Default_Payment

# Check Object Types
fullRaw.info()
    # All are Numerical Columns with no null values

# Check Null Value 
fullRaw.isnull().sum()


#--------------------------------------------------------------------------------------|
# %%

# Event Rate
###############
    # Minority Class should be more than 10%
    # If less than 10%, Data is imbalanced
        # Use SMOTE in such case 

fullRaw["Default_Payment"].value_counts()

# Split 0s & 1s
fullRaw.loc[fullRaw["Source"] == "Train", "Default_Payment"].value_counts() / fullRaw[
                                                fullRaw["Source"] == "Train"].shape[0]
    # Minority Class greater than 10%, that is 0.22
    
    # Another way to do
        # fullRaw.loc[fullRaw["Source"] == "Train", 
        #                               "Default_Payment"].value_counts(normalize=True)

#--------------------------------------------------------------------------------------|
# %%

# Statistical Summary
#######################

fullRaw.describe(include="all").T

# Drop Identifier Column
fullRaw.drop("Customer ID",axis=1,inplace=True)
fullRaw.shape


#--------------------------------------------------------------------------------------|
# %%

# Use Data Description to convert Numerical to Categorical
############################################################

# Categorical Variable:- Gender, Academinc Qualification, Marital

# Gender
variableToUpdate = "Gender"
fullRaw[variableToUpdate].value_counts()
    # 1 indicates Male, 2 indicates female
fullRaw[variableToUpdate].replace({
    1:"Male",
    2:"Female"},inplace=True)
fullRaw[variableToUpdate].value_counts()

# Academic_Qualification
variableToUpdate = "Academic_Qualification"
fullRaw[variableToUpdate].value_counts()
fullRaw[variableToUpdate].replace({
    1:"Undergraduate",
    2:"Graduate",
    3:"Post Graduate",
    4:"Professional",
    5:"Others",
    6:"Unknown"}, inplace=True)
fullRaw[variableToUpdate].value_counts()

# Marital
variableToUpdate = "Marital"
fullRaw[variableToUpdate].value_counts()
fullRaw[variableToUpdate].replace({
    1:"Married",
    2:"Single",
    3:"Unknown", # Donot prefer to say
    0:"Unknown"}, inplace=True)
fullRaw[variableToUpdate].value_counts()


#--------------------------------------------------------------------------------------|
# %%

# Bivariate Analysis - For Continuous Features (BoxPlot)
##########################################################
    # Box Plot because our dependent is categorical

trainDf = fullRaw.loc[fullRaw["Source"] == "Train"]
continuousVars = trainDf.columns[trainDf.dtypes != "object"]
continuousVars

from matplotlib.backends.backend_pdf import PdfPages

fileName = "Continuous Bivariate Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(continuousVars):
    plt.figure(figsize=(10,10))
    plt.tight_layout()
    sns.boxplot(y=trainDf[colName], x=trainDf["Default_Payment"])
    pdf.savefig(colNumber+1) # Page starts from 1 not 0
pdf.close()


#--------------------------------------------------------------------------------------|
# %%
