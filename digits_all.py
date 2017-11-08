
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

dig = load_digits()
X = dig.data 
y = dig.target

np.random.seed(0)
t = len(X)//10
indices = np.random.permutation(len(X))
X_train = X[indices[:-t]]
y_train = y[indices[:-t]]
X_test = X[indices[-t:]]
y_test = y[indices[-t:]]

acc = []
model_names = ['Logistic Regression', 'KNN k=3', 'KNN k=7', 'SVM Linear', 'SVM RBF', 'SVM polynomial', 'CART']


# In[2]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1e5)
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for logistic regression: %d%%\n'%(acc[0]))


# In[3]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for KNN classifier with k=3: %d%%\n'%(acc[1]))


# In[4]:


model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for KNN classifier with k=7: %d%%\n'%(acc[2]))


# In[5]:


from sklearn.svm import SVC
model = SVC(kernel='linear',C=1.0)
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for SVM-linear classifier: %d%%\n'%(acc[3]))


# In[6]:


model = SVC(kernel='rbf',gamma=0.001,C=1.0)
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for SVM-rbf classifier: %d%%\n'%(acc[4]))


# In[7]:


model = SVC(kernel='poly', degree=3, C=1.0)
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for SVM-linear classifier: %d%%\n'%(acc[5]))


# In[8]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
z = model.predict(X_test)
acc.append(accuracy_score(z, y_test)*100)
print(model)
print('Accuracy for CART: %d%%'%(acc[6]))


# In[14]:


import pandas as pd
accuracy = pd.DataFrame({'Classifier':model_names, 'Accuracy':acc})
print(accuracy)


# In[18]:


n_groups = 7
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects = plt.bar(index, acc, bar_width,
                 alpha=opacity,
                 color='b', label='Accuracy')
 
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.xticks(index + bar_width, model_names)
plt.legend()
 
plt.tight_layout()
plt.show()

