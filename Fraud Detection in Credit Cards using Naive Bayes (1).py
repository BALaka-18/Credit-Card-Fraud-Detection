#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


cnt_classes = pd.value_counts(df['Class'], sort = True)
cnt_classes.plot(kind = 'bar', rot = 0)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Transaction Class Distribution")


# In[7]:


df_new = df.sample(frac = 0.1, random_state = 1)
df_new.shape


# In[8]:


# Plotting the frequencies in a histogram for each feature
df_new.hist(figsize = (20,20))
plt.show()


# In[9]:


print(df['Time'].tail(4))
df["Time_Hr"] = df["Time"]/3600
print(df['Time_Hr'].tail(4))


# In[10]:


# Plotting against time to find a trend
plt.hist(df.Time_Hr[df.Class == 0],bins=48,color='g',alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Transactions")
plt.title("Genuine")
plt.show()


# In[11]:


plt.hist(df.Time_Hr[df.Class == 1],bins=48,color='r',alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Transactions")
plt.title("Fraud")
plt.show()


# In[12]:


# Plotting against amount to find a trend
plt.hist(df.Amount[df.Class == 0],bins = 50,color = 'g',alpha=0.5)
plt.xlabel("Amount")
plt.ylabel("Transactions")
plt.title("Fraud")
plt.show()


# In[13]:


plt.hist(df.Amount[df.Class == 1],bins = 50,color = 'r',alpha=0.5)
plt.xlabel("Amount")
plt.ylabel("Transactions")
plt.title("Fraud")
plt.show()


# In[14]:


# Standardizing the Amounts.
from sklearn.preprocessing import StandardScaler
df['Scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'], axis = 1)
df


# In[15]:


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


# In[16]:


def predictions(classifier, x_train, y_train, x_test):
    classifier = classifier   #Creating the classifier
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)
    #Predicted probabilities
    prob = classifier.predict_proba(x_test)
    return predict, prob


# In[17]:


def scores(y_test, predict, prob):
    print("\n1. Confusion matrix : ", confusion_matrix(y_test, predict))
    print("\n2. Recall score : ", recall_score(y_test, predict))
    print("\n3. Accuracy score : ", accuracy_score(y_test, predict))
    print("\n4. Precision score : ", precision_score(y_test, predict))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class_names = df['Class']
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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

