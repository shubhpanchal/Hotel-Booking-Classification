#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Required Libraries
import pandas as pd
import numpy as np
from numpy import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing the dataset
df= pd.read_csv("Copy of Hotel Bookings.csv")
df.head()


# In[3]:


# Checking the info of the data columns and data types
df.info()


# In[4]:


# Checking the cancellations and non cancellations in the dataset
df['is_canceled'].value_counts()


# In[5]:


# Defining our target variable
df['target'] = np.where(df['is_canceled'].isin([1]),1,0)


# In[6]:


# Checking the probability of cancellations
# As the dataset has binary outcomes the probability can be defined by mean of the target variable
df['target'].mean()
# we see that there is 37% cancellations


# In[7]:


# plotting then cancellation count hotelwise
sns.countplot(x='hotel', data = df)
plt.show()


# In[8]:


# Plotting the cancellations and non cancellations in entire dataset
sns.countplot(x='target', data = df)


# In[9]:


# Checking for null values
df.isnull().sum().sort_values(ascending = False)[:5]


# In[10]:


# Treatment of null values
nan_replacements={'children':0,'country':'unknown','agent':0, 'company':0}
df.fillna(nan_replacements,inplace=True)

df['meal'].replace('Undefined','SC', inplace=True)
df['deposit_type'].replace('Refundable', 'Deposits', inplace=True)
df['deposit_type'].replace('Non Refund', 'Deposits', inplace=True)
df['deposit_type'].value_counts()


# In[11]:


# Confirming the treatment of null values
df.isnull().sum().sort_values(ascending=False)[:5]


# In[12]:


# defining the categorical variables
cat=df.describe(include = ['object', 'category']).columns
cat


# In[13]:


# Plotting the cancellation probability with respect to the columns
from numpy import mean
order=['January','February','March','April','May','June','July','August','September','October','November','December']
for col in cat:
    if df[col].nunique()<20:
        plt.figure(figsize=(10,6))
        sns.barplot(x=col, y ='target', data = df, estimator=mean)
        plt.xticks(rotation=45)
        plt.show()
        print('The '+col+'wise cancellation ratios are as follows:')
        print(df.groupby(col)['target'].mean())
        print('_'*90)
    else:
        pass


# In[14]:


# Checking the probability of cancellations in repeated guests
sns.barplot(x='is_repeated_guest', y = 'target', data=df, estimator=mean)
plt.show


# In[15]:


# Replacing the Undefined market segment as Groups market segment 
df['market_segment'].replace('Undefined','Groups', inplace=True)
df.groupby('market_segment')['target'].count()


# In[16]:


# Plotting the cancellations probability with respect to stays_in_weekend_nights
sns.barplot(x='stays_in_weekend_nights', y = 'target', data=df, estimator=mean)


# In[17]:


# Checking the cancellation probabilities with respect to stays_in_weekend_nights
df.groupby('stays_in_weekend_nights')['target'].mean()


# In[18]:


# Plotting the cancellations probability with respect to stays_in_week_nights
sns.barplot(x='stays_in_week_nights', y ='target', data=df, estimator=mean)


# In[19]:


# Dividing the stays_in_week_nights into 3 distict classes using pd.qcut and plotting the cancellation probabilities of classes.
df['stays_in_week_nights_rank'] = pd.qcut(df['stays_in_week_nights'].rank(method='first').values, 3, duplicates='drop').codes+1
sns.barplot(x='stays_in_week_nights_rank', y ='target', data=df, estimator=mean)
print(df.groupby('stays_in_week_nights_rank')['target'].mean())


# In[20]:


# Printing the class boundaries of the classes
print('The Minimum Values in the stays_in_week_nights_rank are :')
print(df.groupby('stays_in_week_nights_rank')['stays_in_week_nights'].min())
print('_'*50)
print('The Maximum Values in stays_in_week_nights_rank are :')
print(df.groupby('stays_in_week_nights_rank')['stays_in_week_nights'].max())


# In[21]:


# Checking the cancellation probabilities with respect to previous_cancellations
sns.barplot(x='previous_cancellations', y ='target', data = df, estimator=mean)


# In[23]:


# Creating a columns with 0 and 1, where 0 represents no prev cancellations
# 1 represents prev cancellations.
# and plotting the cancellation probabilities of each class
df['prev_cancellation_ind']= np.where(df['previous_cancellations'].isin([0]),0,1)
sns.barplot(x=df['prev_cancellation_ind'], y = df['target'], estimator=mean)
plt.show()
print(df.groupby('prev_cancellation_ind')['target'].mean())


# In[24]:


# Replacing Undefined as TA/TO, GDS as Corporate
df['distribution_channel'].replace('Undefined','TA/TO', inplace=True)
df['distribution_channel'].replace('GDS','Corporate', inplace=True)

# Plotting Cancellation probabilities for Distribution Channel
sns.barplot(x=df['distribution_channel'], y = df['target'], estimator=mean)
plt.show()
print('The distribution wise cancellation probabilities are as follows:-')
print(df.groupby('distribution_channel')['target'].mean()) #Printing Cancellation probabilites for distribution channel 
print(' ')
print('The counts of instances:- ')
print(df.groupby('distribution_channel')['target'].count()) #Printing the cancellations count for distribution channel


# In[25]:


# Plotting cancellation probabilities for deposit types
sns.barplot(x=df['deposit_type'], y =df['target'], estimator=mean)
plt.show()
# Printing Cancellation probabilities for deposit type
print(df.groupby('deposit_type')['target'].mean())
print(' ')
# Printing Cancellation Count for Deposit Type
print(df.groupby('deposit_type')['target'].count())


# In[26]:


# replacing Contract as Non-Transient, Group as Non-Transient
df['customer_type'].replace('Contract','Non-Transient', inplace=True)
df['customer_type'].replace('Group','Non-Transient', inplace=True)
# Plotting cancellation probabilities for customer_type
sns.barplot(x=df['customer_type'], y =df['target'], estimator=mean)
plt.show()
# Printing Cancellation Probabilities for customer_type
print(df.groupby('customer_type')['target'].mean())
print(' ')
# Printing cancellation count for customer_type
print(df.groupby('customer_type')['target'].count())


# In[27]:


# Creating a new clolumn that classifies days_in_waiting_list into 3 distinct classes using pd.qcut
# and checking the cancellation probabilites of those classes
df['day_wait_rank']= pd.qcut(df['days_in_waiting_list'].rank(method='first').values,3,duplicates='drop').codes+1
sns.barplot(x=df['day_wait_rank'], y = df['target'], estimator=mean)


# In[56]:


# Printing the cancellation counts in each classes
df.groupby('day_wait_rank')['target'].sum()


# In[29]:


df['day_wait_ind'] = np.where(df['day_wait_rank'].isin([3]),1,0)
sns.barplot(x=df['day_wait_ind'], y = df['target'], estimator=mean)


# In[30]:


# Printing the cancellation count
df.groupby('day_wait_ind')['target'].count()


# In[31]:


# Replacing 3,4 and 5 as 'More than 2' and 1 and 2 as 'Upto 2' and 0 as 'No requests' in total_of_special_requests to create new classes.
df['total_of_special_requests'].replace({3:'More than 2',4:'More than 2',5:'More than 2',
                                        0:'No requests',1:'Upto 2',2:'Upto 2'}, inplace=True)
# Checking the value counts of each class in total_of_special_requests
df['total_of_special_requests'].value_counts()


# In[32]:


# Plotting the cancellation Probabilites for total_of_special_requests
sns.barplot(x=df['total_of_special_requests'], y = df['target'], estimator=mean)


# In[33]:


# Printing the cancellation probabilities for total_of_special_requests
df.groupby('total_of_special_requests')['target'].mean()


# In[34]:


# Cleating a new column where adr is classified into 5 classes and plotting the cancellation probabilities for each class.
df['adr_rank'] = pd.qcut(df['adr'].rank(method='first').values,5, duplicates='drop').codes+1
sns.barplot(x=df['adr_rank'], y =df['target'], estimator=mean)


# In[35]:


# Creating a new column where lead time is classified into 10 distict classes and plotting the cancellation probabilities of each class
df['lead_time_rank'] = pd.qcut(df['lead_time'].rank(method='first').values,10,duplicates='drop').codes+1
sns.barplot(x=df['lead_time_rank'], y = df['target'], estimator=mean)


# In[36]:


df.groupby('lead_time_rank')['target'].sum()


# In[37]:


df.columns


# In[38]:


# Defining the dependent variable
dep = ['target']
# Defining the numerical column that will be used in ML Classification Model
col_num = ['lead_time', 'adr']
# Defining the categorical columns that will be used in ML Classification Model
col_char=['day_wait_ind','total_of_special_requests','customer_type', 'reserved_room_type','distribution_channel',
          'market_segment','prev_cancellation_ind','stays_in_week_nights_rank','deposit_type']


# In[39]:


# Creating the dummy variables for categorical columns
X_char_dum = pd.get_dummies(df[col_char], drop_first=True)


# In[40]:


# Concatenating Categorical and Numerical columns to be Used as X
X_all=pd.concat([df[col_num],X_char_dum], axis=1, join='inner')
X_var=X_all
# Using target as Y
Y_var=df['target']


# In[41]:


# Importing Machine Learning libraries.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[42]:


# Splitting the data into train-test
X_train, X_test, y_train, y_test = train_test_split(X_var, Y_var, test_size=.25, random_state=0)
# Initiating the Logistic regression class
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Initiating the Decision tree class
dtree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=7, min_samples_leaf=10)
dtree.fit(X_train, y_train)
# Initiating the RandomForest Class
rf_1000 = RandomForestClassifier(n_estimators=1000,
                                random_state=2,
                                criterion='gini',
                                max_features='auto',
                                max_depth=7)
rf_1000.fit(X_train,y_train)


# In[43]:


# Evaluating the Accuracy for Linear Regression
y_pred = log_reg.predict(X_test)
print('Accuracy of Logistic Regression on Test Set is : {:.2f}'.format(log_reg.score(X_test, y_test)))


# In[44]:


# Evaluating the Accuracy for DecisionTree
y_pred_tree = dtree.predict(X_test)
print('Accuracy of Decision Tree on Test Data is : {:.2f}'.format(dtree.score(X_test,y_test)))


# In[45]:


# Evaluating the Accuracy for RandomForest
y_pred_forest = rf_1000.predict(X_test)
print('Accuracy of Random Forest on Test Data is : {:.2f}'.format(rf_1000.score(X_test,y_test)))


# In[46]:


# Importing confusion_matrix and classification_report
from sklearn.metrics import confusion_matrix, classification_report
# Calculating confusion_matrix for y_test and y_pred
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[47]:


# Calculating the classification report for decision tree
dtree_cls_rpt = classification_report(y_test, y_pred_tree)
print(dtree_cls_rpt)


# In[48]:


# Printing the classification report for RandomForest
rf_cls_rpt = classification_report(y_test, y_pred_forest)
print(rf_cls_rpt)


# In[49]:


# Importing roc_auc_score androc_curve
from sklearn.metrics import roc_auc_score, roc_curve

logit_roc_auc = roc_auc_score(y_test, log_reg.predict(X_test)) # roc_auc_score for logistic regression
tree_roc_auc = roc_auc_score(y_test, dtree.predict(X_test)) # roc_auc_score for DecisionTree
rf_roc_auc = roc_auc_score(y_test, rf_1000.predict(X_test)) # roc_auc_score for RandomForest

fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:,1]) #roc_curve for logistic regression
fpr, tpr, thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1]) #roc_curve for DecisionTree
fpr, tpr, thresholds = roc_curve(y_test, rf_1000.predict_proba(X_test)[:,1]) # roc_curve for RandomFOrest
# Plotting the roc_curve for Logistic Regression, DecisionTree and RandomForest
plt.figure()
plt.plot(fpr,tpr, label='Logistic Regression(area = %0.2f)' %logit_roc_auc)
plt.plot(fpr,tpr, label='Decesion Tree(area = %0.2f)'%tree_roc_auc)
plt.plot(fpr, tpr, label='Random Forest(area = %0.2f)'%rf_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim(0.0,1.0)
plt.ylim(0.0, 1.05)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# In[50]:


# Saving the logistic regression predict probabilites and dividing into 10 distict classes and calculating mean
y_pred_prob = log_reg.predict_proba(X_var)[:,1]
df['y_pred_P']= pd.DataFrame(y_pred_prob)
df['P_rank']= pd.qcut(df['y_pred_P'].rank(method='first').values,10, duplicates='drop').codes+1
df.groupby('P_rank')['target'].mean()


# In[51]:


# Saving the decisionTree predict probabilites and dividing into 10 distict classes and calculating mean
y_pred_prob_tree = dtree.predict_proba(X_var)[:,1]
df['y_pred_P_tree'] = pd.DataFrame(y_pred_prob_tree)
df['P_rank_tree']= pd.qcut(df['y_pred_P_tree'].rank(method='first').values,10,duplicates='drop').codes+1
df.groupby('P_rank_tree')['target'].mean()


# In[52]:


# Saving the RandomForest predict probabilites and dividing into 10 distict classes and calculating mean
y_pred_prob_rf = rf_1000.predict_proba(X_var)[:,1]
df['y_pred_P_rf'] = pd.DataFrame(y_pred_prob_rf)
df['P_rank_rf']= pd.qcut(df['y_pred_P_rf'].rank(method='first').values,10,duplicates='drop').codes+1
df.groupby('P_rank_rf')['target'].mean()


# In[53]:


df.head()


# In[54]:


# df.to_csv('hotel_demand_prediction_scored_file.csv')


# In[ ]:




