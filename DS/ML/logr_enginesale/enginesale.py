# %%

# Import Libraries
###################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#--------------------------------------------------------------------------------------|


# %%

# Read Dataset
###############

trainDf = pd.read_csv("train_wn75k28.csv")
trainDf["source"] = "train"

testDf = pd.read_csv("test_Wf7sxXF.csv")
testDf["source"] = "prediction"

# Concat
fullRaw = pd.concat([trainDf,testDf],axis=0)
fullRaw.shape

#--------------------------------------------------------------------------------------|

# %%

# Univariate Analysis
######################

fullRaw.head()

fullRaw.info()
    # signup_date, products_purchased null

fullRaw.isna().sum()
    # products_purchased      29047 NaN
    # signup_date             21762 NaN


# Fill NaN
tempMode = fullRaw["products_purchased"].mode()[0]
fullRaw["products_purchased"].fillna(tempMode,inplace=True)

tempMode = fullRaw["signup_date"].mode()[0]
fullRaw["signup_date"].fillna(tempMode,inplace=True)

# Convert Dtypes
def convDtype(col):
    if col == None : 
        return col
    else: 
        return int(col)
fullRaw["products_purchased"].apply(convDtype) # Sometimes astype don't work
fullRaw["buy"] = fullRaw.loc[fullRaw["source"]=="train","buy"].apply(convDtype)

# Handle Date Column
def dateToNum(col):
    num_str = col[:4] + col[5:7] + col[8:]
    return int(num_str)
fullRaw["created_at"] = fullRaw["created_at"].apply(dateToNum)
fullRaw["signup_date"] = fullRaw["signup_date"].apply(dateToNum)

# date_var: consisting of difference between signup & created
fullRaw["date_var"] = fullRaw["signup_date"]-fullRaw["created_at"]

# Drop Identifier
fullRaw.drop(["id"],axis=1, inplace=True)
fullRaw.drop(["created_at","signup_date"],axis=1, inplace=True)

fullRaw.info()

fullRaw.describe().T

#--------------------------------------------------------------------------------------|

# %%

# Train/Test Split
###################

trainDf,testDf = train_test_split(fullRaw[fullRaw["source"]=="train"],random_state=123,     
                                                                     train_size=0.75)
testDf["source"] = "test"    

#--------------------------------------------------------------------------------------|

# %%

# Eventrate
############

trainDf["buy"].value_counts(normalize=True) * 100
    # Data in imbalanced

#--------------------------------------------------------------------------------------|

# %%

# Bivariate Analysis 
######################

# - For Continuous Features (BoxPlot)

continuousVars = ["products_purchased","date_var"]

from matplotlib.backends.backend_pdf import PdfPages
fileName = "Continuous Bivariate Analysis.pdf"
pdf = PdfPages(fileName)
for colNo,col in enumerate(continuousVars):
    plt.figure(figsize=(10,10))
    plt.tight_layout()
    sns.boxplot(x=trainDf["buy"],y=trainDf[col])
    pdf.savefig(colNo+1)
pdf.close()

# - For Categorical Features (HistPlot)

categoricalVars = trainDf.columns
categoricalVars = categoricalVars.drop(["date_var","buy","source",
                                                                "products_purchased"])
fileName = "Bivariate Analysis Categorical.pdf"
pdf = PdfPages(fileName)
for colNo,col in enumerate(categoricalVars):
    plt.figure(figsize=(20,10))
    plt.tight_layout()
    sns.histplot(trainDf,x=col,hue="buy",stat="probability",multiple="stack")
    pdf.savefig(colNo+1)
pdf.close()

#--------------------------------------------------------------------------------------|


# %%

# Dummy Creation
#################

fullRaw = pd.concat([trainDf,testDf],axis=0)
fullRawDf = pd.get_dummies(fullRaw,drop_first=True)
fullRawDf.shape

#--------------------------------------------------------------------------------------|

# %%

# Sampling Dependent & Independent
###################################

train = fullRawDf[fullRawDf["source_train"] == 1].copy()
test = fullRawDf[fullRawDf["source_train"] == 0].copy()

# Drop source_train Column
train.drop("source_train",axis=1,inplace=True)
test.drop("source_train",axis=1,inplace=True)

# X & Y
trainX = train.drop("buy",axis=1).copy()
trainY = train["buy"]

testX = test.drop("buy",axis=1).copy()
testY = test["buy"]
#--------------------------------------------------------------------------------------|

# %%

# Add Intercept Column
########################

from statsmodels.api import add_constant

trainX = add_constant(trainX)
testX = add_constant(testX)

#--------------------------------------------------------------------------------------|

# %%

# VIF Check
############

maxVIF = 10
tempVIF = 10
temp_trainX = trainX.copy()
highVIFcolumnNames = list()

from statsmodels.stats.outliers_influence import variance_inflation_factor

while tempVIF >= maxVIF:

    tempVIFdf = pd.DataFrame()
    tempVIFdf["VIF"] = [variance_inflation_factor(temp_trainX.values ,col) for col in
                                                        range(temp_trainX.shape[1])]
    tempVIFdf["Column"] = temp_trainX.columns
    tempVIFdf.dropna(inplace=True)

    tempVIF = tempVIFdf.sort_values("VIF",ascending=False).iloc[0,0]
    tempColumnName = tempVIFdf.sort_values("VIF",ascending=False).iloc[0,1]

    if tempVIF>=maxVIF:
        highVIFcolumnNames.append(tempColumnName)
        temp_trainX.drop(tempColumnName,axis=1,inplace=True)

highVIFcolumnNames
# No Multi Colinearity
#--------------------------------------------------------------------------------------|


# %%

# Model Building
#################

from statsmodels.api import Logit

Model = Logit(trainY,trainX).fit()
Model.summary()


#--------------------------------------------------------------------------------------|

# %%

# Significant Columns
#####################

maxPval = 0.05
tempPval = 0.05
temp_trainX = trainX.copy()
highPvalColName = list()

while tempPval >= maxPval:

    Model = Logit(trainY,temp_trainX).fit()
    tempPvalDf = pd.DataFrame()
    tempPvalDf["Pval"] = Model.pvalues
    tempPvalDf["Column"] = temp_trainX.columns

    tempPvalDf.dropna(inplace=True)

    tempPval = tempPvalDf.sort_values("Pval",ascending=False).iloc[0,0]
    tempColumnName = tempPvalDf.sort_values("Pval",ascending=False).iloc[0,1]

    if tempPval >= maxPval:
        highPvalColName.append(tempColumnName)
        temp_trainX.drop(tempColumnName,axis=1,inplace=True)

highPvalColName
trainX.drop(highPvalColName,axis=1,inplace=True)
testX.drop(highPvalColName,axis=1,inplace=True)

#--------------------------------------------------------------------------------------|


# %%

# Model Prediction
####################

Model = Logit(trainY,trainX).fit()
testX["Test_Prob"] = Model.predict(testX)

#--------------------------------------------------------------------------------------|


# %%

# Classification
####################

testX["Test_Class"] = np.where(testX["Test_Prob"] >= 0.5,1,0)

#--------------------------------------------------------------------------------------|

# %%

# Model Evaluation
####################

from sklearn.metrics import classification_report

Confusion_Mat = pd.crosstab(testY,testX["Test_Class"])
Confusion_Mat

print(classification_report(testY,testX["Test_Class"]))

#--------------------------------------------------------------------------------------|

# %%

# AUC/ROC Curve
################

from sklearn.metrics import roc_curve, auc

fpr,tpr,cutoff = roc_curve(testY,testX["Test_Prob"])

# Cut Off Table

Cutt_Off_Table = pd.DataFrame()
Cutt_Off_Table["FPR"] = fpr
Cutt_Off_Table["TPR"] = tpr
Cutt_Off_Table["Cutoff"] = cutoff

# Improve Model with Cuttoff

Cutt_Off_Table["Difference"] = Cutt_Off_Table["TPR"] - Cutt_Off_Table["FPR"]
cutt_point = Cutt_Off_Table.sort_values("Difference",ascending=False).iloc[0,2]

testX["Test_Class_2"] = np.where(testX["Test_Prob"] >= cutt_point,1,0)

print(classification_report(testY,testX["Test_Class"]))

#====================================*******============================================|

# %%
