import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

## Load the Data Set
dataset=pd.read_csv("Bank_Personal_Loan_Modelling.csv")
dataset.info()

##Exploratory Data Analysis
##To see all the summary of data let us make one DataFrame that shows feature,dtype,Null value count,Unique count, Unique item.
listitem=[]
for col in dataset.columns:
    listitem.append([col,dataset[col].dtypes,dataset[col].isna().sum(),round((dataset[col].isna().sum()/len(dataset[col]))*100,2),dataset[col].nunique(),dataset[col].unique()])
dfdesc=pd.DataFrame(columns=['Features','dtype','Null Value Count','Null Value Percentage','Unique Count','Unique items'],data=listitem)
dfdesc


 ## Converting to Categorical and Numerical Variables
categorical_val=[]
continuous_val=[]
for column in dataset.columns:   
    if len(dataset[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)

plt.figure(figsize=(17,17))
for i , column in enumerate(categorical_val,1):
    plt.subplot(3,3,i)
    dataset[dataset["PersonalLoan"]==0][column].hist(bins=35,color='red',label='Have Personal Loan = No')
    dataset[dataset["PersonalLoan"]==1][column].hist(bins=35,color='Blue',label="Have Personal Loan = Yes")
    plt.legend()
    plt.xlabel(column)
               
"""
Form the above histogram chart we can see that.
1. Family size of 3 and 4 members are tending to take Personal Loan.
2. Customer that belong to Education category 2 and 3 i.e. Graduate and Professional have taken more Persoanl Loan then the Undergraduate class.
3. Customer who does not have Security Account have taken Personal Loan .
4. Customer who does not CDAcount in this higher number of customer don't have Personal Loan . We can see that customer who have CDAcount most of them had taken Personal Loan. Here CDAccount means Certificate of Deposit.
5. Customer how use Internet Bank service also have higher count of Personal Loan then those who does not use Online Service.
6. Customer who don't have excess to Credit Card for Universal Bank are more likely to apply for PersonaL Loan.
"""


dataset[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14));
##Income, CCAvg , Mortgage have Outlier we will deal with this in Feature Engineering.


sns.pairplot(data=dataset)
"""
From the pair plot we can see that.
1. Age and Experience both have high correlation which each other. 
2. Income,CCAvg,Mortage show positive skewness.
"""
plt.figure(figsize=(7,7))
plt.scatter(x='Age',y='Experience',data=dataset)
##As we can see Age and Experience both have high correlation between each other. We have to remove any one of them in feature Engineering.


##Feature Enginerring
dataset.describe()['Experience']
dataset[dataset['Experience']<0].count()
# Let us replace all the negative Experience data points by absolute value.
dataset['Experience']=dataset['Experience'].apply(abs)
dataset[dataset['Experience']<0].count()
##Now we don't have any negative data points in Experience.


##Now we had outlier in our data set.To treat them we will be replacing all those data points whole value less than equal to LL=(Q1-1.5*IQR) and greater than equal to UL=(Q3+1.5*IQR) by LL and UL.This is known as Capping Method
Outlier = ['Income', 'CCAvg', 'Mortgage']
Q1=dataset[Outlier].quantile(0.25)
Q3=dataset[Outlier].quantile(0.75)
IQR=Q3-Q1

LL,UL = Q1-(IQR*1.5),Q3+(IQR*1.5)

for i in Outlier:
    dataset[i][dataset[i]>UL[i]]=UL[i];dataset[i][dataset[i]<LL[i]]=LL[i] 
    
dataset[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14));
##Now we can see that now we do not have any outlier in data set.



categorical_val.remove('PersonalLoan')
dataset.drop(['ID','Experience','ZIPCode'],axis=1,inplace=True)

##Defining X and y
target_col='PersonalLoan'
X= dataset.loc[:,dataset.columns!=target_col]
y=dataset.loc[:,target_col]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from catboost import CatBoostClassifier


##Initializing Catboost
#let us make the catboost model, use_best_model params will make the model prevent overfitting
model_cb=CatBoostClassifier(iterations=1000, learning_rate=0.01, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)
model_cb.fit(X_train,y_train,cat_features=categorical_val,eval_set=(X_test,y_test))

# prediction
y_pred=model_cb.predict(X_test)

from sklearn.metrics import accuracy_score
accuracyscore=accuracy_score(y_test,y_pred)

# Saving model to disk
# Using pickle 
import pickle
pickle.dump(model_cb,open('model.pkl','wb'))




