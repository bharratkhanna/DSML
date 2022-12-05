#%%

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

#os.chdir("Techstack/ML/property_case/")

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
testDf["Source"] =  "Test"
predictionDf["Source"] =  "Prediction"

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

#%%

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

#%%

##############################################################
# Bivariate Analysis: Analysis between 2 columns: Continuous
############################################################## 

# Continuous Variables:

# Correlation can be observed on multiple columns only, because on a single
# columnn it does not reflect the true data

# * Correlation is only observation/indication and we don't implement resolves
#   on the basis of this

corrDf =  fullRaw[fullRaw["Source"] == "Train"].corr().round(decimals=3)          

# Heatmap
plt.figure(figsize=(20,20))
plot = sns.heatmap(corrDf, 
            xticklabels=(corrDf.columns),
            yticklabels=(corrDf.columns),
            cmap="turbo_r", annot=True) 
plt.tight_layout()
plt.show()            
plt.savefig("plots/heatmap.png")


#%%

###############################################################
# Bivariate Analysis: Analysis between 2 columns: Categorical
###############################################################

categDf = fullRaw.select_dtypes(include="object")
categDf["Sale_Price"] = fullRaw["Sale_Price"][fullRaw["Source"]=="Train"]
categDf.drop("Source",axis=1,inplace=True)

for col in categDf.columns:
    if col != "Sale_Price":
        plt.figure(figsize=(10,15))
        plot = sns.boxplot(y=categDf["Sale_Price"], x=categDf[col])
        plt.savefig("plots/"+col+".png")
        plt.show(block=False)            

#%%
        
#############################################
# Dummy Varialble Creation/ One hot encoding
#############################################

# Dummy Variables: Process of converting categorical columns to a variable that
    #  machine can understand

# n-1 columns are sufficient for giving information to machine about the data

# Alias coefficients / Perfect Multicolinearity: When dummy variables are created 
    # and we don't drop any column.


# Why dummy variables create?
    # - To convert Categorical data to meaningful dat for machine.
    # - To remove perfect multicollinearity
    # - To don't give redundant information to the machine


fullRawDummy = pd.get_dummies(fullRaw, drop_first=True) 
# drop_first to remove one sub category from categorical value to avoid 
    # perfect multicolinearity. Will ensure you get n-1 dummies.
        # - For eg Source with Train,Test,Prediction will come out as 
            # Source_Train, Source_Test

fullRaw.shape
fullRawDummy.shape  # 13 columns were added

########################
# Add Intercept Column
########################

# Because of dummy creation, Model can reccognize its presence but not gather its
    # coefficients, hence to balance this we need constant.

# In python, linear regression does not account for intercept by default.

# So we need to add constant in the dataframe - a column called 
    # as "const" with all values as 1, while applying the regression model, its values 
        # will be updated automatically

# If intercept not added
    # - Coefficients value will increase
    # - All equations will pass from origin
    # - To balance dropped dummies coefficients

from statsmodels.api import add_constant
fullRawDummy = add_constant(fullRawDummy)
fullRawDummy.shape

fullRawDummy.shape

#%% 

#############
# Sampling
#############

# Step1 : Divide into Train,Test and Prediction Data

trainDf = fullRawDummy[fullRawDummy["Source_Train"] == 1].drop(["Source_Test","Source_Train"],axis=1).copy()
testDf = fullRawDummy[fullRawDummy["Source_Test"] == 1].drop(["Source_Train","Source_Test"],axis=1).copy()
predictionDf = fullRawDummy[(fullRawDummy["Source_Train"]== 0 )& (fullRawDummy["Source_Test"]==0)].drop([
    "Source_Train", "Source_Test"
], axis=1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

# Step2: Divide train, test into Dependent & Independent Variables

trainDf.columns

trainX = trainDf.drop("Sale_Price", axis=1).copy()
trainY = trainDf["Sale_Price"].copy()
testX = testDf.drop("Sale_Price", axis=1).copy()
testY = testDf["Sale_Price"].copy()

trainX.shape
trainY.shape
testX.shape
testY.shape
predictionDf.shape

#%%

#############
# VIF Check
#############

from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 # This will be calculated in every iteration of loop
maxVIFCutOff = 5 # 5 is recommended cutoff value for Linear Regression

# Working on temp because we will be removing some columns which we ll check later to 
    # remove or not from orignal trainX
trainXCopy = trainX.copy() 

counter = 1 # Not important
highVIFColumnNames = list() # We'll store High VIF  columns here


while(tempMaxVIF >= maxVIFCutOff):
   
    # Create empty DataFrame to store VIF values
    tempVIFdf = pd.DataFrame()

    # Calculate VIF using List Comprehension
        # trainXCopy.values -> return narray with headers removed.
    tempVIFdf["VIF"] = [variance_inflation_factor(trainXCopy.values, i) for i in range(
        trainXCopy.shape[1])]
    
    # Create new Column_Name to store Columns name against VIF Values
    tempVIFdf["Column_Name"] = trainXCopy.columns

    # Drop NaN if there is some mistake in some calculation resulting in NaN's
    tempVIFdf.dropna(inplace=True)

    # Sort the Columns on basis of VIF values in descending order, and then pick the top 
        # most column name
    tempColumnName = tempVIFdf.sort_values(["VIF"],ascending=False).iloc[0,1]

    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFdf.sort_values(["VIF"],ascending=False).iloc[0,0]

    print(f'{counter}: {tempColumnName}') # Debugger

    if (tempMaxVIF >= maxVIFCutOff): # Check whether its VIF is above 5 or not

        # Remove the highest VIF valued column from trainXCopy
        trainXCopy.drop(tempColumnName,axis=1,inplace=True)
        highVIFColumnNames.append(tempColumnName)

    counter += 1

highVIFColumnNames 
highVIFColumnNames.remove("const") 
highVIFColumnNames

# Removing Columns from orignal Train/Test

trainX.drop(highVIFColumnNames,axis=1,inplace=True)
testX.drop(highVIFColumnNames, axis=1,inplace=True)
predictionDf.drop(highVIFColumnNames, axis=1, inplace=True)

trainX.shape
testX.shape

#%%

##################
# Model Building
##################

from statsmodels.api import OLS

m1ModelDef = OLS(trainY,trainX) # (Dependent,Independent) Defines the model
m1ModelBuild = m1ModelDef.fit() # Building Model
m1ModelBuild.summary() # This is model output summary


# %%
