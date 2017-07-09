
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:

import csv ##
with open("train.csv","rb") as source:
    rdr= csv.reader( source )
    with open("train_res","wb") as result:
        wtr= csv.writer( result )
        for r in rdr:
            wtr.writerow( (r[1], r[4], r[9], r[11]) ) # I choose three features: Sex, Fare, Embarked
df = pd.read_csv('train_res')


# In[3]:

df.loc[df['Sex'] =='male', 'Sex'] = 1.0    #Change Sex values : male-->1 ; female-->0
df.loc[df['Sex'] =='female', 'Sex'] = 0.0
df.loc[df['Embarked'] =='S', 'Embarked'] = 0 # Change Embarked values : S-->0 ; C-->1 ; Q-->2
df.loc[df['Embarked'] =='C', 'Embarked'] = 1
df.loc[df['Embarked'] =='Q', 'Embarked'] = 2
df = df[pd.notnull(df['Embarked'])] #drop the row if the Embarked value is nan
#print(df.to_string())


# In[4]:

sns.pairplot(df,hue='Survived',palette='Set1')


# In[5]:

from sklearn.model_selection import train_test_split
X = df.drop('Survived',axis=1)
y = df['Survived']


# In[7]:

## Train Test Split

#Let's split up the data into a training set and a test set!


# In[8]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[9]:

## Random Forests

#Now let's apply random forest. Note: we don't have to normalize data,
#since it is invariant to monotonic transformations of the features 
#(just think about how the splits are done at each node). 


# In[10]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[11]:

## Prediction and Evaluation 

#Let's evaluate our RF.


# In[12]:

from sklearn.metrics import classification_report,confusion_matrix
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))


# In[13]:

print(classification_report(y_test,rfc_pred))

