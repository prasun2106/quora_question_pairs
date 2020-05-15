#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center"> [Quora Question Pairs Competition on Kaggle](https://www.kaggle.com/c/quora-question-pairs/overview) </div> 
# ## Description
# The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning.
# 
# ### [Data Description](https://www.kaggle.com/c/quora-question-pairs/data) from [competition site](https://www.kaggle.com/c/quora-question-pairs/data):
# The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.
# 
# Please note: as an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. All of the questions in the training set are genuine examples from Quora.
# 
# ### Data fields:
# * `id` - the id of a training set question pair
# * `qid1, qid2` - unique ids of each question (only available in train.csv)
# * `question1, question2` - the full text of each question
# * `is_duplicate` - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.
# 
# ## Approach:
# The approach is explained in the following steps:
# 1. Exploratory Data Analysis
#     - Basic exploration
#     - Missing Values
#     - Outliers
#     - Univariate/Bivariate - if applicable
#     - Feature normaliation
#     - Feature Engineering
# 2. Text Analysis Steps:
#     - Text Processing
#         - Normalization
#             - To lower case
#             - Remove punctuation
#     - Tokenization
#         - Convert it to words
#     - Stopwords removal
#     - Parts of Speech Tagging
#     - Stemming or Lemmatization - choose one of them based on requirement. Sometimes, Lemmatization can take a long time to give results.
#     - Named Entity recognition
# 2. Feature Creation
#     - Create a feature that will indicate the percentage of words common between two questions
# 3. Based on the created feature, train our model to understand the relationship between target variable and the features.
# 
# Without further ado, let's start with importing data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train =  pd.read_csv('train.csv.zip')
df_train.head(3)


# In[3]:


df_test = pd.read_csv('test.csv.zip')
df_test.head(3)


# ## 1. Exploratory Data Analysis

# In[4]:


# Number of question pairs in train and test set
# Number of dupicate pairs in train
# Number of unique questions in the entire train dataset
# Number of duplicate questions in the entire dataset

print(f'shape of train:{df_train.shape}')
print(f'shape of test:{df_test.shape}')
print(f'number of pairs having same questions in train:{df_train.is_duplicate.sum()}')
print(f'number of pairs having different questions in train:{df_train.shape[0] - df_train.is_duplicate.sum()}')
all_questions_train = pd.concat([df_train['question1'],df_train['question2']], axis = 0)
all_questions_train = list(all_questions_train)
all_questions_test = pd.concat([df_test['question1'],df_test['question2']], axis = 0)
all_questions_test = list(all_questions_test)
print(f'total number of questions in train:{len(list(all_questions_train))}')
print(f'total number of questions in test:{len(list(all_questions_test))}')
print(f'total number of unique questions in train:{len(set(all_questions_train))}')
print(f'total number of unique questions in test:{len(set(all_questions_test))}')
print(f'total number of duplicate questions in train:{len(all_questions_train) - len(set(all_questions_train))}')
print(f'total number of duplicate questions in test:{len(all_questions_test) - len(set(all_questions_test))}')


# In[5]:


plt.figure(figsize =  (12,5))
plt.hist(pd.Series(all_questions_train).value_counts(), bins = 50, color = 'black', label = 'train')
plt.hist(pd.Series(all_questions_test).value_counts(), alpha = 0.5, bins = 50, color = 'gray', label = 'test')
plt.yscale('log')
plt.title('Number of occurences of questions')
plt.xlabel('Number of occurences')
plt.ylabel('Number of Questions')
plt.legend()


# There is a large difference in distribution of train set and test set. This might be due to auto-generated questions. From the competition's description:
# 
# > As an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. All of the questions in the training set are genuine examples from Quora.

# ### Missing values

# In[15]:


print(df_train.isna().sum())
print(df_test.isna().sum())


# In[21]:


df_train[df_train['question1'].isna()|df_train['question2'].isna()]


# In[20]:


df_test[df_test['question1'].isna()|df_test['question2'].isna()]


# One interresting thing to note here is that the train and test set both contains few null entries and the questions are all related when there is a null in the second question. In test set, we observe the same question repeating again and again as they are auto-generated with reordering of words.
# 
# Let's drop rows having missing values in train as they are very few. We can't drop rows in test set as we will lose test set ids and then we won't be able to submit our solution. Let's fill test set null with 'not available' for now.

# In[23]:


df_train.dropna(inplace =True)
df_test.replace(np.nan, 'not available', inplace = True)


# In[24]:


# Analyzing number of words
df_train['len_1'] = df_train['question1'].apply(lambda x: len(x.split()))
df_train['len_2'] = df_train['question2'].apply(lambda x: len(x.split()))


# In[27]:


df_test['len_1'] = df_test['question1'].apply(lambda x: len(x.split()))
df_test['len_2'] = df_test['question2'].apply(lambda x: len(x.split()))


# In[ ]:




