#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('car_evaluation.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'decision']
data.columns=col_names


# In[6]:


data.describe()


# In[7]:


#punem in evidenta variabila pe care o vom prezice (variabila dependenta)
import matplotlib.pyplot as plt # vizualizarea datelor
plt.hist(data['decision'])
#observam ca cele mai multe masini sunt neacceptate


# In[8]:


for col in col_names:
    
    print(data[col].value_counts()) 
    
# va trebui sa tranformam toate atributele coloanelor in atribute de tip numeric
# observam ca variabilele sunt distribuite egal pe atribute,cu exceptia variabilei de decizie


# In[9]:


#vedem valorile pe care le poate lua o variabila in functie de coloana
column = data[['buying','maint','doors','persons','lug_boot','safety','decision']]
for x in column:
    print(x," :",set(data[x]))


# In[10]:


#observam ca variabilele sunt de tip obiect, adica categorice
data.info()


# In[11]:


buying_dict = { 'low':1, 'med':2, 'high':3, 'vhigh':4}
maint_dict = {'low':1, 'med':2, 'high':3, 'vhigh':4}
doors_dict = {'2':1, '3':2, '4':3, "5more":4}
persons_dict = {'2':1, '4':2, "more":3}
lug_boot_dict = {'small':1, "med":2, 'big':3}
safety_dict = {'low':1, 'med':2, 'high':3}
decision_dict = {'unacc':0, 'acc':1, 'vgood':1, "good":1}


# In[12]:


data_label=data.copy()
data['buying']=data_label.buying.map(buying_dict)
data['maint']=data_label.maint.map(maint_dict)
data['doors']=data_label.doors.map(doors_dict)
data['persons']=data_label.persons.map(persons_dict)
data['lug_boot']=data_label.lug_boot.map(lug_boot_dict)
data['safety']=data_label.safety.map(safety_dict)
data['decision']=data_label.decision.map(decision_dict)


# In[13]:


data.head()


# In[14]:


#verificam daca sunt valori care lipsesc
data.isnull().sum()


# In[15]:


data.head()


# In[16]:


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[17]:


print(y)


# In[18]:


print(X)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[20]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[21]:


import tensorflow as tf
ann=tf.keras.models.Sequential()


# In[22]:


ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
ann.add(tf.keras.layers.Dense(units=3,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[23]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 500)


# In[24]:


# Evaluate the model
scores = ann.evaluate(X_test, y_test)
print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))


# In[25]:


y_pred=ann.predict(X_test)
y_pred=(y_pred>0.5)


# In[26]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[27]:


acc=accuracy_score(y_test,y_pred)
print(acc)


# In[28]:


new_prediction = ann.predict(sc.transform(np.array([[2, 2, 2, 2, 2, 3]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)


# In[ ]:




