#!/usr/bin/env python
# coding: utf-8

# In[8]:


conda install -c "conda-forge/label/cf202003" cufflinks-py


# In[6]:


import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf
from plotly.offline import init_notebook_mode,plot,iplot

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
import os


# In[7]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[10]:


conda update -n base -c defaults conda


# In[11]:



iris=pd.read_csv(r'C:\Users\andro\Downloads\CodSoft\IRIS DATASET.csv')


# In[14]:


iris


# In[15]:


iris.shape


# In[19]:


px.scatter(iris,x='species',y='petal_width',size='petal_width')


# In[20]:


plt.bar(iris['species'],iris['petal_width'])


# In[21]:


px.bar(iris,x='species',y='petal_width')


# In[22]:


iris.iplot(kind='bar',x=['species'],y=['petal_width'])


# In[23]:


px.line(iris,x='species',y='petal_width')


# In[25]:


px.scatter_matrix(iris,color='species',title='Iris',dimensions=['sepal_length','sepal_width','petal_width','petal_length'])


# In[26]:


iris


# In[28]:


X=iris.drop(['species'],axis=1)


# In[29]:


X


# In[30]:


y=iris['species']


# In[31]:


y


# In[32]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y=le.fit_transform(y)


# In[33]:


y


# In[34]:


X=np.array(X)


# In[35]:


X


# In[36]:


y


# In[37]:



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[38]:


X_train


# In[39]:


X_test.size


# In[41]:


y_train.size


# In[ ]:


#Decision Tree Method


# In[40]:


from sklearn import tree

DT=tree.DecisionTreeClassifier()
DT.fit(X_train,y_train)


# In[42]:


prediction_DT=DT.predict(X_test)
accuracy_DT=accuracy_score(y_test,prediction_DT)*100


# In[43]:


accuracy_DT


# In[44]:


y_test


# In[45]:


prediction_DT


# In[49]:


os.environ["PATH"]+= os.pathsep+(r'C:\Program Files\Graphviz\bin')
import graphviz

vis_data=tree.export_graphviz(DT,out_file=None, feature_names=iris.drop(['Species'],axis=1).keys(),class_names=iris['Species'].unique(),filled=True,rounded=True,special_characters=True)


# In[54]:


os.environ["PATH"]+= os.pathsep+(r'C:\Program Files\Graphviz\bin')
import graphviz

vis_data=tree.export_graphviz(DT,out_file=None, feature_names=iris.drop(['species'],axis=1).keys(),class_names=iris['species'].unique(),filled=True,rounded=True,special_characters=True)


# In[53]:


get_ipython().system('pip install graphviz')


# In[55]:


graphviz.Source(vis_data)


# In[ ]:





# In[56]:


Catagory=['Iris-Setosa','Iris-Versicolor','Iris-Virginica']


# In[ ]:





# In[59]:


X_DT=np.array([[1.4,1.2, 0.6, 1.2]])
X_DT_prediction=DT.predict(X_DT)


# In[60]:


X_DT_prediction[0]
print(Catagory[int(X_DT_prediction[0])])


# In[ ]:





# In[ ]:


#KNN Algorithm


# In[61]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler().fit(X_train)  # Load the standard scaler
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# In[ ]:





# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_std,y_train)


# In[63]:


predict_knn=knn.predict(X_test_std)
accuracy_knn=accuracy_score(y_test,predict_knn)*100


# In[64]:


accuracy_knn


# In[ ]:





# In[65]:


X_knn=np.array([[7.7 ,3.5, 4.6, 4]])
X_knn_std=sc.transform(X_knn)
X_knn_std


# In[66]:


X_knn_prediction=knn.predict(X_knn_std)
X_knn_prediction[0]
print(Catagory[int(X_knn_prediction[0])])


# In[ ]:





# In[67]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[68]:


scores_list


# In[69]:


plt.plot(k_range,scores_list)


# In[ ]:




