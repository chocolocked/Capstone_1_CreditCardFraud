#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:43:37 2017

@author: ginnyzhu
"""
#1.Import libraries 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

from statsmodels.stats.weightstats import ztest
from scipy import stats


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

#z-test for V1-V28 features
#legal vs. fraud
#significant level 99%. Null Hypothesis: Vi fraud = Vi legal 
#any p< 0.10 will reject the null hypothesis 
for col in pca_names:
    p1 =  frauds[col]
    p0 = legals[col]
    z_cal, p_cal= ztest(x1=p1, x2=p0, value=0, alternative='two-sided', usevar='pooled', ddof=1)
    print(col,'statistically significant difference' if p_cal < 0.01 else 'statistically insignificant difference')

#K-S test for time(both Time and hour)
ks_cal1, p_cal1=stats.ks_2samp(frauds.Time, legals.Time)
print('K-S statistics for Time:', ks_cal1, 'p-value:',p_cal1)
ks_cal2, p_cal2=stats.ks_2samp(frauds.hour, legals.hour)
print('K-S statistics for hour:', ks_cal2, 'p-value:',p_cal2)
   
#bootstrapping for amount
#legal vs. fraud
#first define relevant functions:
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D Data"""
    bs_sample = np.random.choice(data.values, len(data))

    return func(bs_sample)

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)

    return bs_replicates

#Hypothesis testing: significant level alpha =0.01, Null Hypthesis: they have the same mean
mean_amount = np.mean(df.Amount.values)
obs_diff_mean = np.mean(frauds.Amount.values) - np.mean(legals.Amount.values)

frauds_shifted = frauds.Amount - np.mean(frauds.Amount.values) + mean_amount
legals_shifted = legals.Amount - np.mean(legals.Amount.values) + mean_amount

bs_replicates_frauds = draw_bs_reps(frauds_shifted, np.mean, 10000)
bs_replicates_legals = draw_bs_reps(legals_shifted, np.mean, 10000)

bs_replicates = bs_replicates_frauds - bs_replicates_legals

# Compute and print p-value: p
p = np.sum(bs_replicates >= obs_diff_mean) / len(bs_replicates)        #because obs_diff_mean= 33.92
print('p-value =', format(p, '.6f'))                                  

sns.distplot(bs_replicates, bins = 50, norm_hist = True, color = 'green')
plt.axvline(x=obs_diff_mean, color='k', linestyle='--')
plt.text(obs_diff_mean-15, 0.0005, 'observed difference')
plt.xlabel('Difference in amount')
plt.ylabel('Percentage')
plt.title('Distribution of Bootstrap replicates of difference (frauds.Amount- legals.Amount)\nUnder H0: they have the same mean')
plt.show()



#4.Prepare Data for Training
