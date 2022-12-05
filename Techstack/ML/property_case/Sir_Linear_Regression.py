# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:32:26 2021

@author: dayanand
"""
# loading library

import os
import pandas as pd
import seaborn as sns

# setting display of screen

pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# loading database

rawDf = pd.read_csv("C:/Users/tk/Desktop/DataScience/Python Class Notes/PropertyPrice_Data.csv") 
predictionDf =  pd.read_csv("C:/Users/tk/Desktop/DataScience/Python Class Notes/PropertyPrice_Prediction.csv") 

rawDf.shape
predictionDf.shape

# sampling the data into train & test

from sklearn.model_selection import train_test_split

trainDf,testDf=train_test_split(rawDf,train_size=0.8,random_state=2410)

trainDf.shape
testDf.shape

# create source column in train,test & prediction datasets

trainDf["source"] = "Train"
testDf["source"] = "Test"
predictionDf["source"] = "Prediction"

# cobine train,test & prediction

fullRaw=pd.concat([trainDf,testDf,predictionDf],axis=0)
# rbind function of R.inpython concat(axis=0(row wise concat))
fullRaw.shape

# drop identifier column i.e id

fullRaw=fullRaw.drop(["Id"],axis=1)
#fullRaw.drop(["Id],axis=1,inplace=True)
fullRaw.columns
#drop is the function to drop the column

#check dtypes of the variable
fullRaw.dtypes# object-categorical,numeric/int-int64&float64

################################
########## Univariate Analysis-Missing value computation
################################

# Missing value check-NaN value check
# isnull() & isna() method to check NA values
fullRaw.isnull().sum()
#fullRaw.isna().sum()

# we have missing values in one categorical variable i.e Garage
#  & one numerical variable i.s Garage_Built_Year

# Missing value computation for categorical variable-Garage

fullRaw.loc[fullRaw["source"]=="Train","Garage"]
tempMode=fullRaw.loc[fullRaw["source"]=="Train","Garage"].mode()[0]
tempMode

 #trainDf["Garage"].mode()[0]

fullRaw["Garage"].fillna(tempMode,inplace=True)
#fullRaw["Garage]=fullRaw["Garage].fillna(tempMode)

# checking NA values in Garage after imputation

fullRaw["Garage"].isna().sum()

#Missing value computation for numeric variable-Garage_Built_Year
fullRaw.columns
fullRaw["Garage_Built_Year"].isna().sum()

tempMedian=fullRaw.loc[fullRaw["source"]=="Train","Garage_Built_Year"].median()
fullRaw["Garage_Built_Year"].fillna(tempMedian,inplace=True)
#fullRaw["Garage_Built_Year]=fullRaw["Garage_Built_Year].fillna(tempMedian)

#Check for all NA's
fullRaw.isna().sum()

# write code to automate NA's value from all columns using for loop

##############
#Bivariate Analysis-Correlation(Scatterplot) for continuous variables
##############

CorrDf=fullRaw[fullRaw["source"]=="Train"].corr()
CorrDf.head()
sns.heatmap(CorrDf,xticklabels=CorrDf.columns,yticklabels=CorrDf.columns,cmap="winter")

# # To know possible names of color palettes, pass an incorrect 
# palette name (cmap argument) in the above code.

# Make correlation plot using some predefine value input

############
## Bivariate Analysis for categorical variable-using boxplot
############
fullRaw.columns
# first option to do itmanualy for each variable when we have less no's of categorical variables

#sns.boxplot(y=fullRaw.loc[fullRaw["source"]=="Train","Sale_Price"],x=fullRaw.loc[fullRaw["source"]=="Train","Road_Type"]) 
sns.boxplot(y=trainDf["Sale_Price"],x=trainDf["Road_Type"])
sns.boxplot(y=trainDf["Sale_Price"],x=trainDf["Property_Shape"])

# 2nd way to do using for loop
Categorical_variables=trainDf.columns[trainDf.dtypes==object]
Categorical_variables
from matplotlib.pyplot import figure
for i in Categorical_variables:
    figure()
    sns.boxplot(y=trainDf["Sale_Price"],x=trainDf[i])

# 3rd way to save plots in pdf

fileName="C:/Users/tk/Desktop/DataScience/Python Class Notes/Categorical_Variable_Analysis.pdf"

from matplotlib.backends.backend_pdf import PdfPages
pdf=PdfPages(fileName)

for colNumber,colName in enumerate(Categorical_variables):
    #enumerate gives keyvalue pair
    print(colNumber,colName)
    figure()
    sns.boxplot(y=trainDf["Sale_Price"],x=trainDf[colName])
    pdf.savefig(colNumber+1)
    #colNumber+1 ensures that page no starts from 1 not o 
pdf.close()
    
