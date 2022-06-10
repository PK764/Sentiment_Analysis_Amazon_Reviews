#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and reading dataset

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()

data=pd.read_csv("C:/Users/konda/Downloads/Reviews.csv1/Reviews.csv",nrows=100)
data


# In[2]:


data.describe() #to see the information of the numerical data


# In[3]:


# Deleting missing values
data.dropna()


# In[4]:


data.isnull().sum()


# In[5]:


data.dropna(axis=1)


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


ratings=data['Score'].value_counts()


# In[9]:


ratings


# In[10]:


numbers=ratings.index
numbers


# In[11]:


quantity=ratings.values
quantity


# In[12]:


plt.pie(quantity,labels=numbers)
plt.title("Product Ratings")


# # Exploring the positive,negative and neutral text in "Text" column

# In[13]:


sentiments=SentimentIntensityAnalyzer() 
# SentimentIntensityAnalyzer is an object and polarity_scores is a function


# In[14]:


sentiments=SentimentIntensityAnalyzer()


# In[15]:


data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Text"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Text"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Text"]]
data.head()


# In[20]:


sum(data["Positive"])


# In[21]:


sum(data["Negative"])


# In[22]:


sum(data["Neutral"])


# In[ ]:


# From this, we can draw a conclusion that positive and neutral scores are more than negative score


# In[ ]:




