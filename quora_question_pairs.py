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
# 2. Text Analysis Steps [NLP]:
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

# In[6]:


print(df_train.isna().sum())
print(df_test.isna().sum())


# In[7]:


df_train[df_train['question1'].isna()|df_train['question2'].isna()]


# In[8]:


df_test[df_test['question1'].isna()|df_test['question2'].isna()]


# One interresting thing to note here is that the train and test set both contains few null entries and the questions are all related when there is a null in the second question. In test set, we observe the same question repeating again and again as they are auto-generated with reordering of words.
# 
# Let's drop rows having missing values in train as they are very few. We can't drop rows in test set as we will lose test set ids and then we won't be able to submit our solution. Let's fill test set null with 'not available' for now.

# In[9]:


df_train.dropna(inplace =True)
df_test.replace(np.nan, 'not available', inplace = True)


# In[10]:


# Analyzing number of words
df_train['len_1'] = df_train['question1'].apply(lambda x: len(x.split()))
df_train['len_2'] = df_train['question2'].apply(lambda x: len(x.split()))


# In[11]:


df_test['len_1'] = df_test['question1'].apply(lambda x: len(x.split()))
df_test['len_2'] = df_test['question2'].apply(lambda x: len(x.split()))


# In[12]:


# plot words distribution for question 1 and question 2 in train set
plt.figure(figsize=(12,4))
plt.hist(df_train['len_1'], label = 'question1', color = 'black', bins = 100)
plt.hist(df_train['len_2'], label = 'question2', color = 'gray', alpha = 0.5, bins =100)
plt.xlabel('number of words')
plt.ylabel('number of questions')
# plt.xlim(0,150)
plt.legend()


# In[13]:


# Comparing differences in number of words between duplicate questions with that of unique questions
plt.figure(figsize = (12,8))
sns.boxplot(x = df_train['is_duplicate'], y = np.abs(df_train['len_1'] - df_train['len_2']) )
plt.xlabel('is_duplicate')
plt.ylabel('min, median, max of number of words')


# From the plot on the left side, we can infer that the number of words are higly different if the qustions are not same. On the other hand, from the plot on right side, we can infer that the number of words in two questons which are same, are very close.

# ## 2. Text Analysis [Preliminary steps of Natural Language Processing]
#    - Text Processing
#         - Normalization
#             - To lower case
#             - Remove punctuation
#    - Tokenization
#         - Convert it to words
#    - Stopwords removal
#    - Parts of Speech Tagging
#    - Stemming or Lemmatization - choose one of them based on requirement. Sometimes, Lemmatization can take a long time to give results.
#    - Named Entity recognition

# In[14]:


# Combine train and test questions only for cleaning purposes. After all the preprocessing, we will separate them
# into two sets
df_train['source'] = 'train'
df_test['source'] = 'test'
train_test = pd.concat([df_train, df_test], axis = 0)


# In[15]:


# Uncomment the following piece of code and download punkt package if not present in your system
# import nltk
# nltk.download()


# In[17]:


# Normalization
import string
punc = string.punctuation
train_test[['question1','question2']] =train_test[['question1','question2']].apply(lambda x: x.apply(lambda y: ''.join(char for char in y.lower() if char not in punc) ))

# Tokenization
from nltk.tokenize import word_tokenize
train_test[['question1','question2']] =train_test[['question1','question2']].apply(lambda x: x.apply(lambda y: word_tokenize(y)))

# Stopwords removal
# from nltk.corpus import stopwords
# train_test[['question1','question2']] =train_test[['question1','question2']].apply(lambda x: x.apply(lambda y: word_tokenize(y)))


# Stopwords removal is taking a long time given the size of our dataset. Let's skip it for now and create the feature based on our current dataset

# In[24]:


# Let's separate training and testing set
train = train_test[train_test['source'] == 'train']
test = train_test[train_test['source'] == 'test']
train.drop(['source','test_id'], axis = 1, inplace = True)
test.drop(['source','id', 'qid1','qid2','is_duplicate'], axis = 1, inplace = True)
print(train.columns)
print(test.columns)


# In[ ]:





# ## Text to Features (feature engineering on text data)
# 

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[26]:


tfidf = TfidfVectorizer()


# In[29]:


corpus =list (train['question1'])


# In[30]:


vector = tfidf.fit_transform(corpus)


# In[35]:


# print(vector)


# In[ ]:




