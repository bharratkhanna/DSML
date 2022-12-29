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

fullRaw = pd.read_csv("Telecom_Churn.csv")

#--------------------------------------------------------------------------------------|

# %%

# Univariate Analysis
######################

fullRaw.head()

fullRaw.info()
    # No Null Values

#--------------------------------------------------------------------------------------|


# %%

# Train/Test Split
####################

trainDf,testDf = train_test_split(fullRaw,random_state=123,train_size=0.7)

trainDf["source"] = "train"
testDf["source"] = "test"

# Combine Train/Test
fullRaw = pd.concat([trainDf,testDf],axis=0)
fullRaw.shape

#--------------------------------------------------------------------------------------|

# %%

# General Analysis
###################

# Recheck NaN
fullRaw.isna().sum()

fullRaw.summary = fullRaw.describe().T

# Remove Identifier 
fullRaw.drop("customerID",axis=1,inplace=True)

#--------------------------------------------------------------------------------------|

# %%

# Label Encoding
##################

fullRaw.head()

fullRaw["Churn"] = np.where(fullRaw["Churn"] == "Yes", 1, 0)
fullRaw["Churn"].value_counts()
#--------------------------------------------------------------------------------------|

# %%

# Event Rate
#############

fullRaw.loc[ fullRaw["source"] == "train" , "Churn"].value_counts(normalize=True) * 100

#--------------------------------------------------------------------------------------|
# %%

# Outlier Detection & Correction
##################################

fullRaw.columns
columnsOutliers = ["tenure","TotalAmount","MonthlyServiceCharges"]

for col in columnsOutliers:
    Q3 = np.percentile(fullRaw.loc[fullRaw["source"]== "train", col],75)
    Q1 = np.percentile(fullRaw.loc[fullRaw["source"]== "train", col],25)
    IQR = Q3-Q1
    UB = Q3 + 1.5*IQR
    LB = Q1 - 1.5*IQR

    # Update Columns
    fullRaw[col] = np.where(fullRaw[col] > UB,UB,fullRaw[col])
    fullRaw[col] = np.where(fullRaw[col] < LB,LB,fullRaw[col])

#--------------------------------------------------------------------------------------|

# %%

# Dummy Variable Creation
###########################

    # Not Dropping because Multi Colinearity not an Issue in ML Models
fullRawDf = pd.get_dummies(fullRaw) 
fullRawDf.shape

#--------------------------------------------------------------------------------------|

# %%

# Dependent/Independent

trainX = fullRawDf[fullRawDf["source_train"]==1].drop(["Churn","source_train",
                                                            "source_test"],axis=1).copy()
trainY = fullRawDf[fullRawDf["source_train"]==1]["Churn"].copy()

testX = fullRawDf[fullRawDf["source_train"]==0].drop(["Churn","source_train",
                                                            "source_test"],axis=1).copy()
testY = fullRawDf[fullRawDf["source_train"]==0]["Churn"].copy()

trainX.shape
testX.shape 

#--------------------------------------------------------------------------------------|

# %%

# Model Building
##################

from sklearn.tree import DecisionTreeClassifier

Model = DecisionTreeClassifier(random_state=123).fit(trainX,trainY)

#--------------------------------------------------------------------------------------|

# %%

# Model Evaluation
###################

Test_Pred = Model.predict(testX)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testY,Test_Pred)
Confusion_Mat

from sklearn.metrics import classification_report

# Classification Report
print(classification_report(testY,Test_Pred))

#--------------------------------------------------------------------------------------|


# %%


# Model Visualization
######################

from sklearn.tree import plot_tree

Model2 = DecisionTreeClassifier(random_state=123, min_samples_leaf = 500).fit(trainX,
                                                                                trainY)
plt.figure(figsize=(20,10))
DT_Plot = plot_tree(Model2, fontsize=10, feature_names=trainX.columns, 
                                                    filled=True,class_names=["0","1"])

#--------------------------------------------------------------------------------------|
# %%
