#%%

# Import Libraries 
#####################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%

# Set Display Option
######################

pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)
pd.set_option("display.width",100000)

# %%

# General Analysis:-
######################


# Read Dataset

train_bfSalesDf = pd.read_csv('train_blackfriday_sales.csv')
predict_bfSaleDf = pd.read_csv('test_blackfriday_sales.csv')


# Glance at a dataset

train_bfSalesDf.head()
predict_bfSaleDf.head()
 # Multiple products bought by same ID's


# Shape of the Dataset

print(train_bfSalesDf.shape)
print(predict_bfSaleDf.shape)
   # Large Dataset, possible in python only


# Info around columns

train_bfSalesDf.info()
 # Categorical - (Could be wrong)
    # Product_Id,Gender,Age,City_Category, Stay_In_Current_City_Years,
 # Nan in Product_Category_3 & Product_Category_2


# Stastical Summary

train_bfSalesDf.describe().T
 # Outliers -
    # Product_Category_1

# %%

# Dropping Identifier
######################

bfSalesDf = pd.concat([train_bfSalesDf,predict_bfSaleDf],axis=0)
bfSalesDf.head()
bfSalesDf.shape

bfSalesDf.drop(["User_ID"],axis=1,inplace=True)
bfSalesDf.head()

# %%

# Imputing NaN's
##################

bfSalesDf.columns

bfSalesDf["Product_Category_2"].fillna(bfSalesDf["Product_Category_2"].median(), inplace=True) 
# bfSalesDf["Product_Category_3"].fillna(bfSalesDf["Product_Category_3"].median(), inplace=True) 
bfSalesDf.drop("Product_Category_3",axis=1,inplace=True)

bfSalesDf.info()
bfSalesDf.head()

# %%

# # Data Type Conversions  
# #########################

def stay_years(col):
   if col == "4+":
      return 4
   else:
      return int(col)

def product_id(col):
   col = col[1:]
   return col

bfSalesDf.loc[bfSalesDf["Gender"] == "F", "Gender"] = 1
bfSalesDf.loc[bfSalesDf["Gender"] == "M", "Gender"] = 0
bfSalesDf["Gender"] = bfSalesDf["Gender"].astype(int)

bfSalesDf["Stay_In_Current_City_Years"] = bfSalesDf["Stay_In_Current_City_Years"].apply(stay_years)
bfSalesDf["Stay_In_Current_City_Years"] = bfSalesDf["Stay_In_Current_City_Years"].astype(int)

bfSalesDf["Product_ID"] = bfSalesDf["Product_ID"].apply(product_id)
bfSalesDf["Product_ID"] = bfSalesDf["Product_ID"].astype(int)
bfSalesDf.head()

# %%

# Bi-variate Analysis : 
########################

analysisDf = bfSalesDf[bfSalesDf["Purchase"].isnull()==False]

# Continuous
plt.figure(figsize=(10,10))
sns.heatmap(bfSalesDf.corr().round(decimals=2), vmin=-1,vmax=1,annot=True)

plt.figure(figsize=(10,10))
analysisDf2 = analysisDf.drop("Product_ID",axis=1).copy()
sns.pairplot(analysisDf2)
plt.tight_layout()

objectDf = analysisDf.select_dtypes(include=object)
objectDf["Purchase"] = bfSalesDf[bfSalesDf["Purchase"].isnull()==False]["Purchase"]

# Categoriacal
for col in objectDf.columns:
   
   if col != "Purchase":
      plt.figure(figsize=(10,10))
      sns.boxplot(y = objectDf["Purchase"], x = objectDf[col])
      plt.show()

for col in analysisDf.columns: # ["Occupation", "Product_Category_1", "Product_Category_2"] or ["Gender","Stay_In_Currect_City_Years",City_Category,Age]
   for hue in analysisDf.columns: #["Gender","Stay_In_Currect_City_Years",City_Category,Age]
      if hue != col and hue != "Product_ID" and col != "Product_ID":
         plt.figure(figsize=(10,10))
         sns.barplot(y=analysisDf["Purchase"],x=analysisDf[col],
                                                      hue=analysisDf[hue])
         plt.show()

# %%

# Outliers
###########

def outlier(col,u_bound,l_bound):
   if col < l_bound:
      return l_bound
   elif col > u_bound:
      return u_bound
   else:
      return col

for col in bfSalesDf.columns:
   if bfSalesDf[col].dtype != "object":
       IQR = bfSalesDf[col].quantile(0.75) - bfSalesDf[col].quantile(0.25)
       u_bound_cal = bfSalesDf[col].quantile(0.75) + (1.5* IQR)
       l_bound_cal = bfSalesDf[col].quantile(0.25) - (1.5* IQR)
       bfSalesDf[col] = bfSalesDf[col].apply(outlier, args=(u_bound_cal,l_bound_cal))

bfSalesDf.describe().T 


# %%

# Scaling
############

def norm(col,max_val,min_val):
   col = (col - min_val) / (max_val-min_val)
   return col

max_val_id = max(bfSalesDf["Product_ID"])
min_val_id = min(bfSalesDf["Product_ID"])
bfSalesDf["Product_ID"] = bfSalesDf["Product_ID"].apply(norm,args=(max_val_id,min_val_id))


#%% 

# Dummies Creation
#####################

bfSalesDf = pd.get_dummies(bfSalesDf,drop_first=True)

# %%

# Add Intercept
##################

from statsmodels.api import add_constant
bfSalesDf =  add_constant(bfSalesDf)
bfSalesDf.shape


# %%

# Test/Train/Split
###################

trainDf,testDf = train_test_split(bfSalesDf, train_size=0.85, random_state=13)
trainDf.shape
testDf.shape

trainDf = trainDf[trainDf["Purchase"].isnull()==False]
testDf = testDf[testDf["Purchase"].isnull()==False]
predictionDf = bfSalesDf[bfSalesDf["Purchase"].isnull()==True]

predictionDf.shape
trainDf.shape
testDf.shape

# %%

# Sampling- Independet/Dependent
#################################

trainDfX = trainDf.drop("Purchase",axis=1).copy()
trainDfY = trainDf["Purchase"].copy()

testDfX = testDf.drop("Purchase", axis=1).copy()
testDfY = testDf["Purchase"].copy()


# %%

# VIF Check
############

from statsmodels.stats.outliers_influence import variance_inflation_factor

maxVIFCuttoff = 5
tempMaxVIF = 5
tempTrainDfX = trainDfX.copy()
highVIFColumnNames = list()

while(tempMaxVIF >= maxVIFCuttoff):

   tempVIFDf = pd.DataFrame()
   tempVIFDf["VIF"] = [variance_inflation_factor(tempTrainDfX.values, col) for col in
                                                      range(tempTrainDfX.shape[1])]
   tempVIFDf["Column_Name"] = tempTrainDfX.columns

   tempVIFDf.dropna(inplace=True)

   tempMaxVIF = tempVIFDf.sort_values("VIF",ascending=False).iloc[0,0]
   tempMaxVIFColumnName = tempVIFDf.sort_values("VIF",ascending=False).iloc[0,1]

   if tempMaxVIF >= maxVIFCuttoff:
      tempTrainDfX.drop(tempMaxVIFColumnName,axis=1,inplace=True)
      highVIFColumnNames.append(tempMaxVIFColumnName)

      

highVIFColumnNames
highVIFColumnNames.remove('const')

trainDfX.drop(highVIFColumnNames,axis=1,inplace=True)
testDfX.drop(highVIFColumnNames,axis=1,inplace=True)
predictionDf.drop(highVIFColumnNames,axis=1,inplace=True)

trainDfX.shape
testDfX.shape
predictionDf.shape
# %%

# Model Building 
#################

from statsmodels.api import OLS

tempModel = OLS(trainDfY,trainDfX).fit()
tempModel.summary()

# %%

# Evaluation
##############

Model = OLS(trainDfY,trainDfX).fit()
Model.summary()

test_pred = Model.predict(testDfX)

# MSE
(np.mean( (testDfY-test_pred)**2 )) # Result is shit
# MAPE
np.mean(np.abs( (testDfY-test_pred))/testDfY ) * 100

testDfX

# %%

# BLUE Check
###############

# Homoskedadticity
plt.figure(figsize=(10,10))
sns.scatterplot(Model.fittedvalues, Model.resid)

# Normalization of Residuals
plt.figure(figsize=(10,10))
sns.distplot(Model.resid)


# Does not satisfy Blue, either Model should be reevaluated or There is no
#  significant relation between Independent & dependent features