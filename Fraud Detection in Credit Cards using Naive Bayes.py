#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, recall_score, precision_score, accuracy_score, f1_score


# In[2]:


df = pd.read_csv("C:\\Users\\BIPLAB\\Desktop\\creditcard.csv")
df


# In[3]:


# shape
print(df.shape)


# In[4]:


print(df.head(7))


# In[5]:


# check null values exist or not
print(df.info())


# In[6]:


print(df['Time'].tail(4))
df["Time_Hr"] = df["Time"]/3600
print(df['Time_Hr'].tail(4))


# In[7]:


# Plotting against time to find a trend
plt.hist(df.Time_Hr[df.Class == 0],bins=48,color='g',alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Transactions")
plt.title("Genuine")
plt.show()


# In[8]:


plt.hist(df.Time_Hr[df.Class == 1],bins=48,color='r',alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Transactions")
plt.title("Fraud")
plt.show()


# In[9]:


# Plotting against amount to find a trend
plt.hist(df.Amount[df.Class == 0],bins = 50,color = 'g',alpha=0.5)
plt.xlabel("Amount")
plt.ylabel("Transactions")
plt.title("Fraud")
plt.show()


# In[10]:


plt.hist(df.Amount[df.Class == 1],bins = 50,color = 'r',alpha=0.5)
plt.xlabel("Amount")
plt.ylabel("Transactions")
plt.title("Fraud")
plt.show()


# In[11]:


# Standardizing the Amounts.
from sklearn.preprocessing import StandardScaler
df['Scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'], axis = 1)
df


# In[12]:


def split(df, dropped):
    df = df.drop(dropped, axis = 1)
    print(df.columns)
    # Train Test splitting
    from sklearn.model_selection import train_test_split
    y = df['Class']
    x = df.drop(['Class'], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 40, stratify = y)
    
    print("\nTrain size : ", len(y_train), "\nTest size : ", len(y_test))
    return x_train, x_test, y_train, y_test

# Check
drop = []
split(df, drop)


# In[13]:


def predictions(classifier, x_train, y_train, x_test):
    classifier = classifier   #Creating the classifier
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)
    #Predicted probabilities
    prob = classifier.predict_proba(x_test)
    return predict, prob


# In[14]:


def scores(y_test, predict, prob):
    print("\n1. Confusion matrix : ", confusion_matrix(y_test, predict))
    print("\n2. Recall score : ", recall_score(y_test, predict))
    print("\n3. Accuracy score : ", accuracy_score(y_test, predict))
    print("\n4. Precision score : ", precision_score(y_test, predict))


# In[21]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#Doing the actual prediction with Gaussian Naive Bayes

#Case 1 : Dropping none of the columns
dropped = []
X_train, X_test, Y_train, Y_test = split(df, dropped)
y_pred, y_prob = predictions(GaussianNB(), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[16]:


#Case 2 : We are dropping some important parameters of the PCA components
dropped = ['V6', 'V28', 'V27', 'V16', 'V18', 'V9', 'V10', 'V25', 'V20', 'V17', 'V11']
X_train, X_test, Y_train, Y_test = split(df, dropped)
y_pred, y_prob = predictions(GaussianNB(), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[17]:


#Case 3 : We are dropping some important parameters of the PCA components and Time too
dropped = ['Time_Hr', 'V6', 'V28', 'V27', 'V16', 'V18', 'V9', 'V10', 'V25', 'V20', 'V17', 'V11', 'V8', 'V5']
X_train, X_test, Y_train, Y_test = split(df, dropped)
y_pred, y_prob = predictions(GaussianNB(), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[18]:


#Case 4 : We are dropping some important parameters of the PCA components + Time + Scaled_Amount
dropped = ['Time_Hr', 'Scaled_Amount', 'V6', 'V27', 'V16', 'V18', 'V9', 'V10', 'V25', 'V20', 'V17', 'V11']
X_train, X_test, Y_train, Y_test = split(df, dropped)
y_pred, y_prob = predictions(GaussianNB(), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[24]:


#Doing the actual prediction with Logistic Regression
#Case 1 : Dropping none of the columns
dropd = []
X_train, X_test, Y_train, Y_test = split(df, dropd)
y_pred, y_prob = predictions(LogisticRegression(penalty = 'l1', C = 0.001), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[25]:


#Case 2 : We are dropping some important parameters of the PCA components
dropd = ['V6', 'V28', 'V27', 'V16', 'V18', 'V9', 'V10', 'V25', 'V20', 'V17', 'V11', 'V8', 'V12', 'V21', 'V22']
X_train, X_test, Y_train, Y_test = split(df, dropd)
y_pred, y_prob = predictions(LogisticRegression(penalty = 'l1', C = 0.001), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[26]:


#Case 3 : We are dropping some important parameters of the PCA components + Time + Scaled_Amount
dropd = ['Time_Hr', 'V6', 'V28', 'V27', 'V16', 'V18', 'V9', 'V10', 'V25', 'V20', 'V17', 'V11', 'V8', 'V5']
X_train, X_test, Y_train, Y_test = split(df, dropd)
y_pred, y_prob = predictions(LogisticRegression(penalty = 'l1', C = 0.001), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[28]:


#Case 4 : We are dropping some important parameters of the PCA components + Time + Scaled_Amount
dropd = ['Time_Hr', 'Scaled_Amount', 'V6', 'V27', 'V16', 'V18', 'V9', 'V10', 'V25', 'V20', 'V17', 'V11', 'V5', 'V8', 'V21', 'V22']
X_train, X_test, Y_train, Y_test = split(df, dropd)
y_pred, y_prob = predictions(LogisticRegression(penalty = 'l1', C = 0.001), X_train, Y_train, X_test)
print("\n",y_pred, y_prob)
scores(Y_test, y_pred, y_prob)


# In[31]:


# We see performance on imbalanced dataset is poor
# So we train the model on 50/50 under-sampled data, i.e, we take 50/50 ratio of both the classes

# Step 1: Get indices of fraud and genuine data
f_ind = np.array(df[df.Class == 1].index)
g_ind = df[df.Class == 0].index

# Total number of fraud cases
fraud = len(df[df.Class == 1])

# Step 2: Select randomly from genuine class
rand_gen = np.array(np.random.choice(g_ind, fraud, replace = False))

# Step 3: Merging the two class indices : random genuine + original fraud
under_sample = np.concatenate([f_ind, rand_gen])

# Step 4: Creating the undersampled dataset and separating features and target data
new_df = df.iloc[under_sample,:]     # Creating the under sample dataset
y_df = new_df['Class'].values        # Label/Target
x_df = new_df.drop(['Class'], axis = 1).values      # Features

# Step 5: Some information extraction
print("\nTransactions in undersampled data : ", len(new_df))
print("\nPercetage genuine transactions : ", int((len(new_df[new_df.Class == 0])/len(new_df))*100), "%")
print("\nPercetage fraud transactions : ", int((len(new_df[new_df.Class == 1])/len(new_df))*100), "%")


# In[32]:


# Now we train the Logistic Regression model on the undersampled dataset

drpd = []
X_train_new, X_test_new, Y_train_new, Y_test_new = split(new_df, drpd)
y_pred_new, y_pred_prob_new = predictions(LogisticRegression(C = 0.01, penalty = 'l1'), X_train_new, Y_train_new, X_test_new)
print("\n", y_pred_new, y_pred_prob_new)
scores(Y_test_new, y_pred_new, y_pred_prob_new)


# In[34]:


# Now let us train on the undersampled dataset and test on the entire dataset
Y_total = df['Class'].values     # Label/Target
X_total = df.drop(['Class'], axis = 1).values     # Features

logr = LogisticRegression(C = 0.002, penalty = 'l1')
logr.fit(x_df, y_df)
p = logr.predict(X_total)

print("\nConfusion matrix : \n", confusion_matrix(Y_total, p)) 
print("\nPrecision score : ", precision_score(Y_total, p))
print("\nAccuracy score : ", accuracy_score(Y_total, p))


# In[35]:


'''Hence we see that Logistic Regression gives better model sensitivity, whereas predictive value(positive) is more for 
   Gaussian Naive Bayes. '''

