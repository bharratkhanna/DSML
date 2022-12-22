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

# Bivariate Analysis - For Categorical Features (Histogram)
############################################################

categoricalVars = trainDf.columns[trainDf.dtypes == "object"]
categoricalVars

fileName = "Categorical Bivariate Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(categoricalVars):
    if colName == "Source": continue
    plt.figure(figsize=(10,10))
    plt.tight_layout()
    sns.histplot(trainDf, x=colName, hue="Default_Payment", stat="probability",
                                                                     multiple="fill")
    pdf.savefig(colNumber+1)
pdf.close()

#--------------------------------------------------------------------------------------|

# %%

# Dummy Creation
##################

fullRawDf = pd.get_dummies(fullRaw, drop_first=True)
    # Avoiding Perfect Multicolinearity
fullRawDf.shape

#--------------------------------------------------------------------------------------|

# %%

# Sampling Dependent & Independent
####################################

fullRawDf.columns

train = fullRawDf[fullRawDf["Source_Train"] == 1].copy()
test = fullRawDf[fullRawDf["Source_Train"] == 0].copy()

# Drop Source_Train Column
train.drop("Source_Train",axis=1,inplace=True)
test.drop("Source_Train",axis=1,inplace=True)

# Divide into X & Y
trainX = train.drop("Default_Payment",axis=1).copy()
trainY = train["Default_Payment"].copy()

testX = test.drop("Default_Payment",axis=1).copy()
testY = test["Default_Payment"]

trainX.shape
testX.shape
 
#--------------------------------------------------------------------------------------|

# %%

# Add Intercept Column
#######################

from statsmodels.api import add_constant

trainX = add_constant(trainX)
testX = add_constant(testX)

trainX.shape
testX.shape

#--------------------------------------------------------------------------------------|

# %%

# VIF Check
############

from statsmodels.stats.outliers_influence import variance_inflation_factor

maxVIF = 10 # In Logistic Regression, MultiColinearity is not a big issue
tempVIF = 10
highVIFColNames = list()
tempTrainX = trainX.copy()

while(tempVIF >= maxVIF):

    tempVIFDf = pd.DataFrame()
    
    tempVIFDf["VIF"] = [variance_inflation_factor(tempTrainX.values,colNum)
                                            for colNum in range(tempTrainX.shape[1])]
    tempVIFDf["Column_Name"] = tempTrainX.columns

    tempVIFDf.dropna(inplace=True)

    tempVIF = tempVIFDf.sort_values("VIF",ascending=False).iloc[0,0]
    tempColName = tempVIFDf.sort_values("VIF",ascending=False).iloc[0,1]

    if tempVIF >= maxVIF:
        tempTrainX.drop(tempColName,axis=1,inplace=True)
        highVIFColNames.append(tempColName)

highVIFColNames

highVIFColNames.remove("const")
trainX.drop(highVIFColNames,axis=1,inplace=True)
testX.drop(highVIFColNames,axis=1,inplace=True)

trainX.shape
testX.shape

#--------------------------------------------------------------------------------------|                                                    
# %%

# Model Building
#################

from statsmodels.api import Logit

Model = Logit(trainY,trainX).fit()
Model.summary()

    # Converged: True
    # Refers to the point where model is able to find best set of values for
    # coefficients

#--------------------------------------------------------------------------------------|
# %%

# Significant Columns
#######################

tempPval = 0.05
maxPval = 0.05
tempTrainX = trainX.copy()
highPvalColumns = list()

while tempPval >= maxPval:

    Model = Logit(trainY,tempTrainX).fit()

    tempPvalDf = pd.DataFrame()

    tempPvalDf["Pval"] = Model.pvalues
    tempPvalDf["Column_Name"] = tempTrainX.columns

    tempPvalDf.dropna(inplace=True)

    tempPval = tempPvalDf.sort_values("Pval",ascending=False).iloc[0,0]
    tempColName = tempPvalDf.sort_values("Pval",ascending=False).iloc[0,1]

    if tempPval >= maxPval:
        tempTrainX.drop(tempColName,axis=1,inplace=True)
        highPvalColumns.append(tempColName)


highPvalColumns
trainX.drop(highPvalColumns,axis=1,inplace=True)
testX.drop(highPvalColumns,axis=1,inplace=True)
trainX.shape


#--------------------------------------------------------------------------------------|

# %%

# Model Prediction
###################

Model = Logit(trainY,trainX).fit()

testX["Test_Prob"] = Model.predict(testX) # Stored value in Probability
testX["Test_Prob"][:6]
testY[:6]

#--------------------------------------------------------------------------------------|

# %%

# Classification
#################

testX["Test_Class"] = np.where(testX["Test_Prob"] >= 0.5,1,0)
testX["Test_Class"][:6]


#--------------------------------------------------------------------------------------|

# %%

# Model Evaluation
####################


# Confusion Matrix
Confusion_Mat = pd.crosstab(testX["Test_Class"],testY) # R<C Format
Confusion_Mat

# Accuracy Check - Manual
sum(np.diagonal(Confusion_Mat)) *100 / testX.shape[0]

# Sklearn Classification Report
from sklearn.metrics import classification_report
print(classification_report(testY,testX["Test_Class"]))

    # Support -> No of rows for category
    # Macro Avg -> Sum of Categories Average on specific Test like 0.84+0.66/2
    # Weighted Average = 0.84*7052 + 0.66*1948 / 9000 (Multiply Rows too)

    # Precision/Recall.... consider 0 as TP when calculated for 0 & 1 as TP for 1
    # F1 Score is low for Defaulter, hence model is not beneficial for bank
    
#--------------------------------------------------------------------------------------|

# %%

# AUC/ROC Curve
#################

from sklearn.metrics import roc_curve,auc

# Predict on train data
Train_Prob = Model.predict(trainX)
fpr,tpr, cutoff = roc_curve(trainY,Train_Prob)

# Cut off Table Creation
    # Best Cutt off is High TPR & Low FPR
Cutoff_Table = pd.DataFrame()
Cutoff_Table["FPR"] = fpr
Cutoff_Table["TPR"] = tpr
Cutoff_Table["Cuttoff"] = cutoff
Cutoff_Table.shape

# Plot cutoff_table 
plt.figure(figsize=(20,10))
sns.lineplot(x=Cutoff_Table["FPR"],y=Cutoff_Table["TPR"])
plt.tight_layout() 

# Area under Curve
auc(fpr,tpr)

#--------------------------------------------------------------------------------------|

# %%

# Imporve Model with Cut Off Point
###################################

Cutoff_Table["Difference"] = Cutoff_Table["TPR"] - Cutoff_Table["FPR"]

# Euclidean Distance
    # Highest TPR is 1
    # Lowest FPR is 2
Cutoff_Table["Distance"] = np.sqrt( (1-Cutoff_Table["TPR"])**2 +
                                                             (0-Cutoff_Table["FPR"])**2 )

# Most Appropriate Cutoff Point
cutoffPoint = Cutoff_Table.sort_values("Difference",ascending=False).iloc[0,2]

# Classify Model
testX["Test_Class2"] = np.where(testX["Test_Prob"] >= cutoffPoint, 1, 0)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testX["Test_Class2"],testY)
Confusion_Mat

print(classification_report(testY,testX["Test_Class2"]))

#======================================================================================|