#!/usr/bin/env python
# coding: utf-8

# In[4]:


import lightgbm as lgb
import pandas as pd
import os
os.chdir('C://Users//Kyle//Desktop//MPCS_Fall_2019//Advanced_Data_Analytics//project//face recognition')
related_df_1 = pd.read_csv('auto_dist_related_v2.csv', names=["auto_euclidean", "auto_cosine"])
related_df_2 = pd.read_csv('facenet_dist_related.csv', names=["facenet_euclidean", "facenet_cosine"])
related_df_3 = pd.read_csv('vgg_dist_related.csv', names=["vgg_euclidean", "vgg_cosine"])


# In[6]:


related_df = pd.concat([related_df_1, related_df_2, related_df_3], axis=1, sort=False)
related_df.shape


# In[7]:


related_label = [1] * related_df.shape[0]
related_df['label'] = related_label
related_df.shape


# In[8]:


unrelated_df_1 = pd.read_csv('auto_dist_unrelated_v2.csv', names=["auto_euclidean", "auto_cosine"])
unrelated_df_2 = pd.read_csv('facenet_dist_unrelated.csv', names=["facenet_euclidean", "facenet_cosine"])
unrelated_df_3 = pd.read_csv('vgg_dist_unrelated.csv', names=["vgg_euclidean", "vgg_cosine"])
unrelated_df = pd.concat([unrelated_df_1, unrelated_df_2, unrelated_df_3], axis=1, sort=False)
unrelated_label = [0] * unrelated_df.shape[0]
unrelated_df['label'] = unrelated_label
unrelated_df.shape


# In[9]:


train_df = pd.concat([related_df, unrelated_df], axis=0, sort=False)
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.iloc[:3]


# In[10]:


x = train_df.drop(columns=['label'])
y = train_df['label']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)


# In[11]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 2,
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 64,
    'verbose': 2
}
model = lgb.train(params
                  , train_data
                  , valid_sets=test_data, early_stopping_rounds=50
                  , num_boost_round=500 
                 )


# In[12]:


test_df_1 = pd.read_csv('auto_dist_test_v2.csv', names=["auto_euclidean", "auto_cosine"])
test_df_2 = pd.read_csv('facenet_dist_test.csv', names=["facenet_euclidean", "facenet_cosine"])
test_df_3 = pd.read_csv('vgg_dist_test.csv', names=["vgg_euclidean", "vgg_cosine"])
test_df = pd.concat([test_df_1, test_df_2, test_df_3], axis=1, sort=False)
test_df.shape


# In[13]:


y_pred = model.predict(test_df)


# In[15]:


y_submit = [m[1] for m in y_pred]
y_submit[:10]


# In[17]:


sub_df = pd.read_csv('sample_submission.csv')
sub_df.is_related = y_submit
sub_df.to_csv("submission.csv", index=False)


# In[ ]:




