
# coding: utf-8

# In[1]:

## Import Libraries


# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
get_ipython().magic(u'matplotlib inline')


# In[2]:

## Get the Data


# In[3]:

with open("train.csv","rb") as source:
    rdr= csv.reader( source )
    with open("train_res","wb") as result:
        wtr= csv.writer( result )
        for r in rdr:
            wtr.writerow( (r[1], r[4], r[9], r[11]) ) # I choose three features: Sex, Fare, Embarked
df = pd.read_csv('train_res')
df.head(5)


# In[4]:

## Data PreProcessing
## the normalization of feature vectors prior to feeding them to the SVM is very 
## important see "http://neerajkumar.org/writings/svm/"


# In[5]:

df.loc[df['Sex'] =='male', 'Sex'] = 1    #Change Sex values : male-->1 ; female-->0
df.loc[df['Sex'] =='female', 'Sex'] = -1
df.loc[df['Embarked'] =='S', 'Embarked'] = -1 # Change Embarked values : S-->-1 ; C-->0 ; Q-->1
df.loc[df['Embarked'] =='C', 'Embarked'] = 0
df.loc[df['Embarked'] =='Q', 'Embarked'] = 1
df = df[pd.notnull(df['Embarked'])] #drop the row if the Embarked value is nan
#print(df.to_string())


# In[6]:

#Data normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaler.fit(df['Fare'])


# In[7]:

NorFare=scaler.transform(df['Fare'])


# In[8]:

df['Fare']=NorFare


# In[9]:

df.head(10)


# In[10]:

## Train Test Split


# In[11]:

from sklearn.model_selection import train_test_split
X = df.drop('Survived',axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[12]:

# Train the Support Vector Classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)


# In[13]:

## Predictions and Evaluations

#Now let's predict using the trained model.


# In[14]:

predictions = model.predict(X_test)


# In[17]:

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[18]:

print(classification_report(y_test,predictions))

