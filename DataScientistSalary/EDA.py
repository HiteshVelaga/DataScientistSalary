# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:20:04 2020

@author: Hitesh
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('cleaned_data.csv')

def title(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'other'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'other'
    
df['job_simp'] = df['Job Title'].apply(title)
df.job_simp.value_counts()

df['seniority'] = df['Job Title'].apply(seniority)
df.seniority.value_counts()

# Fix state Los Angeles 
df['State']= df.State.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
df['State']= df.State.apply(lambda x: x.strip() if x.strip().lower() != 'la' else 'CA')
df.State.value_counts() 

df['num_comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)

df['At_Headquaters'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)
df.Rating.hist()
#there are outliers with -1 value

df.avg_salary.hist()
#it is pretty much a normal distribution

df.age.hist()
#it is right skewed
df.columns
df_cat = df[['Location', 'Headquarters', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 'State','At_Headquaters', 'python', 'R',
       'spark', 'aws', 'excel', 'job_simp', 'seniority']]

for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()

pd.pivot_table(df, index = 'job_simp', values = 'avg_salary')
pd.pivot_table(df, index = ['job_simp','seniority'], values = 'avg_salary')
pd.pivot_table(df, index = ['State','job_simp','seniority'], values = 'avg_salary').sort_values('State', ascending = False)

pd.pivot_table(df[df.job_simp == 'data scientist'], index = 'State', values = 'avg_salary').sort_values('avg_salary', ascending = False)

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = " ".join(df['Job Description'])

def punctuation_stop(text):
    """remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered


words_filtered = punctuation_stop(words)

text = " ".join([ele for ele in words_filtered])

wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
wc.generate(text)

plt.figure(figsize=[10,10])
plt.imshow(wc,interpolation="bilinear")
plt.axis('off')
plt.show()

df.to_csv("eda.csv")
