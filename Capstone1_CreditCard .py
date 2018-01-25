#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:43:37 2017

@author: ginnyzhu
"""
#1.Import libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#2.Import data
fn='creditcard.csv'
df = pd.read_csv(fn)

#3.1 Summary stats for the entire dataset 
#checking missing data 
df.isnull().any().sum()  #O
print(df.info())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.head(5))
print(df.tail(5))

#Visualize correlations between all variables
sns.heatmap(df.corr(),cmap = 'Greys')
plt.xticks(rotation=60)
plt.yticks(rotation=60)
plt.show()

#By different categories: PCA features, Time, Amount, and Class(fraud or not)
#Visual EDA, PCA features plot
df_pca = df.iloc[:,1:29]
df_pca.plot(kind='Box')
plt.xticks(rotation=60)
plt.title('Boxplots for V1-V28')
plt.xlabel('Features from PCA')
plt.ylabel('Values')
plt.show()
plt.savefig('Boxplots for V1-V28.png')

#EDA and visual EDA for time
print(df['Time'].describe())
print(len(df['Time'].value_counts())) 
plt.figure()
sns.distplot(df['Time'], bins = 200, norm_hist = True, color = 'black') 
plt.title('Distribution of transaction time elapsed since 1st transaction')
plt.show()
plt.savefig('time.png')
 
#Convert time into datetime obejct, is it necessary?
#Also it seems that python crashed easily here
from datetime import datetime, timedelta
def GetTime(time):
    sec = timedelta(seconds=time)
    d = datetime(1,1,1) + sec
    #("DAYS:HOURS:MIN:SEC")
    return d 
    #return "%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second)      
df_newtime = [GetTime(time) for ind, time in df['Time'].iteritems()]
df['Newtime'] = df_newtime 
#Extract hour information
df['hour'] = [(GetTime(time)).hour for ind, time in df['Time'].iteritems()]

#EDA and visual EDA for amount 
print(df['Amount'].describe())
plt.figure()
df.plot(y='Amount', kind='box')
plt.title('Boxplot for transaction amout $')
plt.ylabel('Transaction Amount $')
plt.show()
plt.savefig('Amount_box.png')

#Class ==1:fraud. Confirm the number of frauds in the dataset.(Expected: Inbalanced)
fnum = df['Class'].sum()
fperc = float(fnum)/len(df['Class'])
print(fperc)


#3.2 EDA comparing fraud vs. legal transactions 
frauds=df[df['Class']==1]
legals=df[df['Class']==0]

#V1-V28, boxplots for fraud and legal transactions 
frauds.iloc[:,1:29].plot(kind='Box', color = 'red')
plt.xticks(rotation=60)
plt.title('Boxplots for fraud V1-V28')
plt.xlabel('Features from PCA')
plt.ylabel('Values')
plt.show()
plt.savefig('Boxplots for fraud V1-V28.png')
legals.iloc[:,1:29].plot(kind='Box', color = 'blue')
plt.xticks(rotation=60)
plt.title('Boxplots for legal V1-V28')
plt.xlabel('Features from PCA')
plt.ylabel('Values')
plt.show()
plt.savefig('Boxplots for legal V1-V28.png')
#Iterate through the 28 features and compare distibution for each fraud&legal pair
pca_names = df_pca.columns
print(pca_names)
for i, V in enumerate(df[pca_names]):
    plt.figure(i+1)
    sns.distplot(frauds[V], bins = 50, norm_hist = True, color = 'red')
    sns.distplot(legals[V], bins = 50, norm_hist = True, color = 'blue')
    plt.title('Distribution of fraud vs.legal:feature: ' + str(V))
    plt.show()
    plt.savefig(str(V)+'.png')
    plt.close()

#Time, distribution plots for fraud vs. legal 
plt.figure()
frauds.Time.plot(kind='hist', bins=100, color = 'red', normed= True,label= 'Fraud', alpha = 0.3)
legals.Time.plot(kind='hist', bins=100, color = 'blue', normed= True, label= 'Legal',alpha = 0.3)
plt.legend(loc='upper right')
plt.title('Histogram of transaction time elapsed since 1st transaction: Fraud vs. Legal')
plt.xlabel('Time elapsed since 1st transaction (seconds)')
plt.ylabel('Percentage')
plt.show()
plt.savefig('Fraud vs. Legal Transaction Time.png')
#Check the hours 
plt.figure()
frauds.hour.plot(kind='hist', bins=100, color = 'red', normed= True,label= 'Fraud', alpha = 0.3)
legals.hour.plot(kind='hist', bins=100, color = 'blue', normed= True, label= 'Legal',alpha = 0.3)
plt.legend(loc='upper right')
plt.title('Histogram of transaction hour of the day elapsed since 1st transaction: Fraud vs. Legal')
plt.xlabel('hour since 1st transaction')
plt.ylabel('Percentage')
plt.show()
plt.savefig('Fraud vs. Legal Transaction hour.png')

#Amount,summary stats and plots for fraud vs. legal 
print(frauds.Amount.describe())
print(legals.Amount.describe())
plt.figure()
plt.subplot(1,2,1)
frauds.Amount.plot(kind='Box',color = 'red')
plt.title('Frauds')
plt.subplot(1,2,2)
legals.Amount.plot(kind='Box',color = 'blue')
plt.title('Legal transactions')
plt.ylabel('Transaction amount $')
plt.tight_layout()
plt.show()
plt.savefig('Frauds vs. Legal.png')

#Amount Vs time(Optional, could be discarded)
n0=plt.scatter(legals.Time, legals.Amount,color='blue', label='Legal', alpha = 0.3)
n1=plt.scatter(frauds.Time, frauds.Amount,color='red', label='Fraud', alpha = 0.3)
#n1 drawn later will produce a better graph 
plt.legend((n1, n0),('Fraud', 'Legal'),scatterpoints=1, loc='upper left', ncol=1, fontsize=8)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Amount vs time')
plt.rcParams['font.size']=10
plt.show()

#t-test for V1-V28 features
#legal vs. fraud
#significant level: null hypothesis (look for right terms later:and how to intepret stats)
from scipy import stats
for col in pca_names:
    sample =  frauds[col]
    pop = legals[col]
    print(col, stats.ttest_ind(sample,pop))

#t-test for amount
sample = frauds.Amount
pop  = legals.Amount
print(stats.ttest_ind(sample,pop,equal_var = False))


#4.Prepare Data for Training
