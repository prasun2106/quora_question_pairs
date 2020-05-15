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

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train =  pd.read_csv('../input/quora-question-pairs/train.csv.zip')
df_train.head(3)


# In[ ]:


df_test = pd.read_csv('../input/quora-question-pairs/test.csv.zip')
df_test.head(3)


# ## 1. Exploratory Data Analysis

# In[ ]:


# Number of question pairs in train and test set
# Number of dupicate pairs in train
# Number of unique questions in the entire dataset

print(f'shape of train:{df_train.shape}')
print(f'shape of test:{df_test.shape}')
print(f'number of duplicate pairs in train:{df_train.is_duplicate.sum()}')
all_questions_train = pd.concat([df_train['question1'],df_train['question2']], axis = 0)
print(f'toal number of questions in train:{len(all)}')



qid = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

print(f'The total number of question pairs in the training set: {len(df_train)}')
print(f"Duplicate Pairs: {sum(df_train['is_duplicate'])}")
print(f"Number of unique questions: {df_train['qid1'].append(df_train['qid2']).nunique()}")
#print(f"Number of repeating questions: {len(df_train['qid1'].append(df_train['qid2'])) - df_train['qid1'].append(df_train['qid2']).nunique()}")

print(f'Number of repeating questions: {np.sum(qid.value_counts() > 1)}')


# Plotting the histogram of number repeating questions


# In[ ]:


qid.value_counts()


# In[ ]:


# plotting the histogram

plt.figure(figsize =  (12,5))

plt.hist(qid.value_counts(), bins = 50)
plt.yscale('log')
plt.title('Number of occurences of questions')
plt.xlabel('Number of occurences')
plt.ylabel('Number of Questions')


# # Test Submission
# 

# In[ ]:


from sklearn.metrics import log_loss


# In[ ]:


p = df_train['is_duplicate'].mean()
print('predicted score:', log_loss(y_true= df_train['is_duplicate'], y_pred =  np.zeros(len(df_train['is_duplicate'])) + p))


# In[ ]:


# Submission

df_test = pd.read_csv('../input/quora-question-pairs/test.csv')


# In[ ]:


df_train


# In[ ]:


df_test.head()


# In[ ]:


submission = df_test.copy()


# In[ ]:


# submission['is_duplicate'] = p


# In[ ]:


submission.drop(['question1','question2'],axis =1, inplace = True)


# In[ ]:


submission.reset_index(drop = True)


# In[ ]:


submission.head()


# In[ ]:


#submission.to_csv('baseline.csv',index = False)


# Leaderboard score - 0.55  
# Predicted score - 0.65

# # Text Analysis

# In[ ]:


df_train.info()


# # Missing Values

# In[ ]:


df_train.isna().sum()


# In[ ]:


df_train[df_train['question1'].isna() | df_train['question2'].isna()]


# In[ ]:


df_train.dropna(inplace = True)


# In[ ]:


df_train.info()


# In[ ]:


df_train['question1_len'] = df_train['question1'].apply(len)
df_train['question2_len'] = df_train['question2'].apply(len)


# In[ ]:


df_train.head()


# In[ ]:


# Appending both question 1 and 2 together
question_train = pd.Series(list(df_train['question1']) + list(df_train['question2']))
question_test = pd.Series(list(df_test['question1'])  + list (df_test['question2']))

# finding length of questions
question_train_length = pd.Series(question_train.apply(lambda x: len(str(x).split())))
question_test_length = pd.Series(question_test.apply(lambda x: len(str(x).split())))


# In[ ]:


#plotting the length of question 1 and 2 from the training set
plt.figure(figsize = (17,8))
plt.xlim(0,200)
sns.distplot(question_train_length,bins = 200, norm_hist = True, label = 'train') 
sns.distplot(question_test_length, bins = 200, norm_hist = True, label = 'test')


# In[ ]:


train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)


# In[ ]:


train_qs


# In[ ]:


#Histogram

plt.figure (figsize = (15,10));

plt.hist(dist_train, bins = 200, range = [0,200], color = [0.4, 0.3, 0.1], label = 'Train', normed = True);
plt.hist(dist_test, bins = 200, range = [0,200], color = [0.2,0.8,0.1], label = 'Test',normed = True, alpha = 0.7);

plt.title('Normalized histogram of Character Count in training and testing set questions');
plt.xlabel('Number of Words');
plt.ylabel('Normalised frequency (Probability)');
plt.legend()


# In[ ]:


len('djnfjdfj jnsndjsfn')
len('jdjf jnvjn'.split(' '))


# In[ ]:


train_word = train_qs.apply(lambda x: x.split(' '))
test_word = test_qs.apply(lambda x : x.split(' '))
n_word_train = train_word.apply(len)
n_word_test = test_word.apply(len)


# In[ ]:


# Plotting Histogram of word count

plt.figure(figsize = (15,10))

plt.hist(n_word_train, bins = 50, range = [0,100], color = [0.9,0.3,0.1],density = True, label = 'Train');
plt.hist(n_word_test, bins = 50, range = [0,100], color = [0.0,1,0], density = True, alpha = 0.5, label = 'Test' );

plt.legend()
plt.title('Histogram of number of words in training and testing sets')
plt.xlabel('Number of words')
plt.ylabel('Normalized frequency (Probability)')


# Distribution for both training and testing set is almost same. Let's look at the most common words.

# In[ ]:


duplicate = df_train.groupby('is_duplicate')
duplicate[['question1_len','question2_len']].mean()


# In[ ]:


df_train['question1_word'] = df_train['question1'].apply(lambda x : len(x.split(' ')))
df_train['question2_word'] = df_train['question2'].apply(lambda x : len(x.split(' ')))


# In[ ]:


duplicate[['question1_word','question2_word']].mean()


# From the above two averages, we can conclude that number of words and number of characters are lesser in the duplicate pairs

# In[ ]:


# Word Cloud

# from wordcloud import WordCloud

# word_cloud = WordCloud(width = 800, height = 400, max_words = 250, background_color= 'White', colormap = 'Blues').generate(" ".join(train_qs.astype(str)))


# In[ ]:


# plt.figure(figsize = (25,20))

# plt.imshow(word_cloud)


# # Steps of NLP:
# 
# For finding the matches between both the questions we will have to create a suitable feature.
# 
# Our first feature is going to be word-match-share.
# 
# 1. Initial Feature Analysis

# 1. training set - question 1  
# for a row - i will find the total number of unique words:
# 1. i will find the exact common words ratio (after removing stopwords)
# 2. I will find the common lemma or roots
# 3. In both the cases, the distribution of duplicate will be plotted
# 
# Steps:
# 1. remove punctuation
# 2. make them small letters
# 3. remove stopwords
# 4. stemming

# ### First Step:
# For finding the duplicate questions, we will check the ratio of similar lemma to the total number of unique lemma after removing the stopwords.
# Few transformations will be applied to each questions:
# 1. Normalization
#     * Lowercase
#     * Punctuation removal
# 2. Tokenization
# 3. Stopwords Removal
# 4. Lemmatization
# 
# ### Second Step:
# After these steps we can attach parts of speec tag to each word in both the questions. In the first step we checked the ratio of similar lemma based only on its face value. In the second step (the modification), we will check the similar words ratio on the basis of the lemma as well as the POS tag attached to it.

# In[ ]:


question1 = df_train['question1']
question2 = df_train['question2']


# In[ ]:


# removing punctuation and converting to lower case
import string


# In[ ]:


question1 = question1.apply((lambda x: (''.join(c for c in x if c not in string.punctuation)).lower()))
question2 = question2.apply((lambda x: (''.join(c for c in x if c not in string.punctuation)).lower()))


# In[ ]:


# Tokenization:
from nltk.tokenize import word_tokenize


# In[ ]:


import time


# In[ ]:


start  = time.time()
question1 = question1.apply(lambda x: word_tokenize(x))
end = time.time()
print(f'time taken = {start - end}')


# In[ ]:


# start  = time.time()
# question1_map =pd.Series(list(map(word_tokenize,question1)))
# end = time.time()
# print(f'time taken = {start - end}')


# In[ ]:


# Stopwords removal
from nltk.corpus import stopwords


# In[ ]:


question1_stopwords = question1.apply(lambda x: (c for c in x if c not in stopwords.words('english')))


# In[ ]:


question1_stopwords = list(map(lambda x: [word for word in x if word not in stopwords.words('english')], question1))


# In[ ]:


question2_stopwords = list(map(lambda x: [word for word in x if word not in stopwords.words('english')], question2))


# In[ ]:


# Lemmatization
from nltk import pos_tag
# Due to the amount of data, pos tagging is taking a long time. So we are skipping it for now


# In[ ]:


# question1_pos = question1_stopwords.apply(lambda x : pos_tag(x))


# In[ ]:


# question2_pos = question2_stopwords.apply(lambda x : pos_tag(x))


# In[ ]:


from nltk.stem import PorterStemmer


# In[ ]:


stemmer = PorterStemmer()


# In[ ]:


list(map(stemmer.stem, ))


# In[ ]:




