# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:19:13 2020

@author: Hitesh
"""

import pandas as pd 

data=pd.read_csv("glassdoor.csv")

#salary parsing 
data=data[data['Salary Estimate']!='-1']
data['hourly'] = data['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
data['employer_provided'] = data['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)
data=data[data['hourly']!=1]
data=data[data['employer_provided']!=1]
salary=data['Salary Estimate'].apply(lambda x:x.split()[0])
salary = salary.apply(lambda x: x.lower().replace('k','000').replace('$',''))
salary= salary.apply(lambda x: x.lower().replace('(employer',''))
data['min_salary'] = salary.apply(lambda x: int(x.split('-')[0]))
data['max_salary'] = salary.apply(lambda x: int(x.split('-')[1]))
data['avg_salary'] = (data.min_salary+data.max_salary)/2

#Company name 
data['company_txt']=data.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

#state
data['State']=data['Location'].apply(lambda x: x.split(',')[1])

#At_Headquaters
data['At_Headquaters'] = data.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

#age of company 
data['age'] = data.Founded.apply(lambda x: x if x <1 else 2020 - x)

#parsing of job description (python, etc.)

#python
data['python'] = data['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
 
#r studio 
data['R'] = data['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

#spark 
data['spark'] = data['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws 
data['aws'] = data['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel
data['excel'] = data['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

df_out = data.drop(['Unnamed: 0'], axis =1)

df_out.to_csv('cleaned_data.csv',index = False)
