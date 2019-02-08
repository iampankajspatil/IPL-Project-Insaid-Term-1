#!/usr/bin/env python
# coding: utf-8

# <img src = 'https://raw.githubusercontent.com/iampankajspatil/IPL-Project-Insaid-Term-1/07fa971970b3078837d877d1bd5a6ad275eb293b/Indian_Premier_League_IPL.png' width=300, height=300>
# 
# # Indian Premier League
#    ------

# ### Table of Contents
# - 1. [Problem Statement](#pb)</br>
#     - 1.1 [Introduction](#intro)<br/>
#     - 1.2 [Data source and data set](#section102)<br/>
# - 2. [Load the packages and data](#section2)</br>
# - 3. [Data profiling](#section3)</br>
#     - 3.1 [Understanding the Dataset](#section301)<br/>
#     - 3.2 [Pre Profiling](#section302)<br/>
#     - 3.3 [Preprocessing](#section303)<br/>
#     - 3.4 [Post Profiling](#section304)<br/>
# - 4. [Questions](#section4)
#     - 4.1 [How many seasons we’ve got in the dataset?](#section401)<br/>
#     - 4.2 [Which Team had won by maximum runs?](#section402)<br/>
#     - 4.3 [Which Team had won by maximum wicket?](#section403)<br/>
#     - 4.4 [Which Season had most number of matches?](#section404)<br/>
#     - 4.5 [Which IPL Team is more successful?](#section405)<br/>
#     - 4.6 [Which player has won maximum player of the match award?](#section406)<br/>
#     - 4.7 [Has Toss-winning helped in winning matches?](#section407)<br/>
#     - 4.8 [IPL matches played in which stadiums?](#section408)<br/>
#     - 4.9 [Which Team is best in defending the total?](#section409)<br/>
#     - 4.10 [Which Team is best in chasing the target?](#section410)<br/>
#     - 4.11 [Which is the best venue to defend the total?](#section411)<br/>
#     - 4.12 [Which is the best venue to chase the target?](#section412)<br/>
#     - 4.13 [Venue wise result!](#section413)<br/>
#     - 4.14 [Which player received the maximum number of Player of Match award while defending the total?](#section414)<br/>
#     - 4.15 [Which player received the maximum number of Player of Match award while chasing the target?](#section415)<br/>
#     - 4.16 [Player of the Match - Match wise result](#section416)<br/>
#     - 4.17 [Venue & Player of The Match wise results!](#section417)<br/>
# - 5. [Conclusion](#section5)<br/>
# 
# 

# <a id="pb"></a>
# ### 1. Problem Statement
# "This dataset is from a 2018 iplt20.com that shows all the matches with their results of all IPL seasons played from year 2008 till 2018."
# 
# Using his dataset we are going to analyis our following things:-
# 
# 1. How many seasons we’ve got in the dataset?
# 
# 2. Which Team had won by maximum runs?
# 
# 3. Which Team had won by maximum wicket?
# 
# 4. Which Season had most number of matches?
# 
# 5. Which IPL Team is more successful?
# 
# 6. Which player has won maximum player of the match award?
# 
# 7. Has Toss-winning helped in winning matches?
# 
# 8. IPL matches played in which stadiums?
# 
# 9. Which Team is best in defending the total?
# 
# 10. Which Team is best in chasing the target?
# 
# 11. Which the best venue to defend the total?
# 
# 12. Which the best venue to chase the target?
# 
# 13. Venue wise result!
# 
# 14. Which player received the maximum number of Player of Match award while defending the total?
# 
# 15. Which player received the maximum number of Player of Match award while chasing the target?
# 
# 16. Player of The Match - Match wise results!
# 
# 17. Venue & Player of The Match wise results!

# <a id=intro></a> 
# ### 1.1. Introduction
# 
# **The Indian Premier League (IPL)**, officially Vivo Indian Premier League for sponsorship reasons, is a professional Twenty20 cricket league in India contested during April and May of every year by teams representing Indian cities and some states.The league was founded by the **Board of Control for Cricket in India (BCCI)** in 2008, and is regarded as the brainchild of Lalit Modi, the founder and former commissioner of the league. IPL has an exclusive window in ICC Future Tours Programme.
# 
# The IPL is the most-attended cricket league in the world and in 2014 ranked sixth by average attendance among all sports leagues.In 2010, the IPL became the first sporting event in the world to be broadcast live on YouTube.**The brand value of IPL in 2018 was (US dollar 6.3 billion), according to Duff & Phelps. According to BCCI, the 2015 IPL season contributed ₹11.5 billion (US dollar 182 million) to the GDP of the Indian economy.**
# 
# There have been eleven seasons of the IPL tournament. The current IPL title holders are the Chennai Super Kings, who won the 2018 season. The most successful franchises in the tournament are the **Chennai Super Kings** and **Mumbai Indians** with 3 tournament wins each. **(*Source: Wikipedia)**
# 

# <a id=section2></a> 
# ### 2. Load the packages and data 

# In[1]:


# Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling
import matplotlib as mat
import matplotlib.pyplot as plt
import bokeh


# #### Importing the Dataset

# In[2]:


ipl = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Projects/matches.csv')
ipl# Importing ipl dataset using pd.read_csv


# - The dataset consists of the information about IPL tournament held in India during month of April - May. Various variables present in the dataset includes data of teams, venues, toss winers, winners etc. 
# - The dataset comprises of __696 observations of 18 columns__. 

# | Column Name     | Description                                               |
# | -------------   |:-------------                                            :| 
# | season          | Year when tournament is held                              | 
# | city            | In which city match was played                            |  
# | Date            | Match date                                                | 
# | team1           | First team name                                           |   
# | team2           | Second team name                                          |
# | toss_winner     | Which team won the toss                                   |
# | toss_decision   | Number of sibling and/or spouse travelling with passenger |
# | result          | Number of parent and/or children travelling with passenger|
# | dl_applied      | Duckworth–Lewis method used for results                   |
# | winner          | Which team won the match                                  |
# | win_by_runs     | which team defended the total                             |
# | win_by_wickets  | Which team chase the traget                               |
# | player_of_match | Which player played the best to contribute more to win    |
# | venue           | On which ground match was played                          |
# | umpire1         | First standing Umpire name                                |
# | umpire2         | Second standing Umpire name                               |
# | umpire3         | Third Umpire name                                         |

# <a id=section3></a>

# ## 3. Data Profiling

# - In the upcoming sections we will first __understand our dataset__ using various pandas functionalities.
# - Then with the help of __pandas profiling__ we will find which columns of our dataset need preprocessing.
# - In __preprocessing__ we will deal with erronous and missing values of columns. 
# - Again we will do __pandas profiling__ to see how preprocessing have transformed our dataset.

# <a id=section301></a>

# ### 3.1 Understanding the Dataset

# In[3]:


ipl.shape                                       # This will print the number of rows and columns of the Data Frame


# IPL data has __891 rows__ and __12 columns.__

# In[4]:


ipl.info()                                              # This will give Index, Datatype and Memory information


# In[5]:


ipl.count()                                     # This will show missing values in each columns


# In[6]:


ipl.index                                         #This will show start & stop point of Index


# In[7]:


ipl.columns                                       #This will show column's name


# In[8]:


ipl.head(10)                                    #This will show to 10 entries of the dataset


# In[9]:


ipl.describe(include = 'all')                 #This will show distribution of complete dataset


# In[10]:


ipl.sample(10)                          # This will show random values from dataset


# In[11]:


ipl.isnull().sum()


# From the above output we can see that umpire3 columns contains maximum null values.

# <a id=section302></a>

# ### 3.2 Pre Profiling

# In[12]:


profile = pandas_profiling.ProfileReport(ipl)
profile.to_file(outputfile="ipl.html")


# - By pandas profiling, an __interactive HTML report__ gets generated which contins all the information about the columns of the dataset, like the __counts and type__ of each _column_. Detailed information about each column, __coorelation between different columns__ and a sample of dataset.<br/>
# - It gives us __visual interpretation__ of each column in the data.
# - _Spread of the data_ can be better understood by the distribution plot. 
# - _Grannular level_ analysis of each column.

# Here, we have done Pandas Profiling before preprocessing our dataset, so we have named the html file as __ipl.html__. 

# <a id=section303></a>

# ### 3.3 Preprocessing

# - Dealing with missing values<br/>
#     - Dropping the column __'umpire3 & dl_applied'__ as it has too many _null_ & zero values.

# In[13]:


ipl.drop(['umpire3', 'dl_applied'], axis=1, inplace=True)         #Deleting umpire3 & dl_applied columns


# <a id=section304></a>

# ## 3.4 Post Pandas Profiling

# In[14]:


profile = pandas_profiling.ProfileReport(ipl)
profile.to_file(outputfile="ipl_postprofilling.html")


# The pandas profiling report which we have generated after preprocessing will give us more beneficial insights. We can compare the two reports, i.e __ipl.html__ and ipl_postprofilling.html.<br/>
# In titanic_after_preprocessing.html report, observations:
# - In the Dataset info, Total __Missing(%)__ = __0.1%__ 
# - Number of __variables__ = __16__ 

# <a id=section4></a>

# ### 4. Questions

# <a id=section401></a>

# ### 4.1 How many seasons we’ve got in the dataset ?

# In[15]:


ipl['season'].unique()


# We have got all the seasons __Starting from 2008 till last year i.e 2018.__ To know the exact count:

# In[16]:


len(ipl['season'].unique())


# We have got total __11 seasons__.

# <a id=section402></a>

# ### 4.2 Which Team had won by maximum runs?

# In[17]:


ipl.iloc[ipl['win_by_runs'].idxmax()]


# As of now all the match played in entire history of IPL, we have got a match which is played in 2017 season between **Mumbai Indians & Delhi Daredevils** where **Mumbai Indians defended the score & won the match by margin of 146 runs.**

# <a id=section403></a>

# ### 4.2 Which Team had won by maximum wicket?

# In[18]:


ipl.iloc[ipl['win_by_wickets'].idxmax()]


# As of now all the match played in entire history of IPL, we have got a match which is played in 2017 season between **Gujrat Lions & Kolkata Knight Riders** where **Kolkata Night Riders chased the score without lossing a wicket**.

# <a id=section404></a>

# ### 4.4 Which Season had most number of matches?

# In[19]:


sns.countplot(x='season', data=ipl)
plt.show()


# From the above graph we can see that in **the season of 2013 year total 76 matches** were played which is the most among all the season.

# <a id=section405></a>

# ### 4.5 Which IPL Team is more successful?

# In[20]:


data = ipl.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h');


# From the above grapth we can see that **Mumbai Indians** is most succesfull team among all the ipl team played in the history of IPL.

# <a id=section406></a>

# ### 4.6 Which player has won maximum player of the match award?

# In[21]:


top_players = ipl.player_of_match.value_counts()[:10]
fig, ax = plt.subplots()
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
sns.barplot(x = top_players.index, y = top_players, orient='v'); #palette="Blues");
plt.show()


# From the above graph we can see that **Chris Gayle** has won **maximum numbers of Player of The Match** awards

# <a id=section407></a>

# ### 4.7 Has Toss-winning helped in winning matches?

# In[22]:


ss = ipl['toss_winner'] == ipl['winner']
ss.groupby(ss).size()
sns.countplot(ss);


# This result shows us **Toss winning has not affected end match result** as result has shown almost same sount.

# <a id=section408></a>

# ### 4.8 IPL matches played in which stadiums?

# In[23]:


ipl['venue'].unique()


# <a id=section409></a>

# ### 4.9 Which Team is best in defending the total?

# In[24]:


ipl[ipl['win_by_runs']>0].groupby(['winner'])['win_by_runs'].count().sort_values(ascending = False)


# From the above result we can see that **Mumbai Indians has defended the total 51 times** which is most among all other teams, followed by **Chennai Super Kings 49** times & **Kings XI Punjab** 35 times.

# In[25]:


#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Runs - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_runs', data=ipl[ipl['win_by_runs']>0], orient = 'h'); #palette="Blues");
plt.show()


# <a id=section410></a>

# ### 4.10 Which Team is best in chasing the target?

# In[26]:


ipl[ipl['win_by_wickets']>0].groupby(['winner'])['win_by_wickets'].count().sort_values(ascending = False)


# From the above results we can see that **Kolkata Knight Riders as chased the traget 52 times** which is most by any team in IPL history, followed by **Royal Challengers Bangalore & Mumbai Indians 46 times each**. 

# In[27]:


#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Wickets - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_wickets', data=ipl[ipl['win_by_wickets']>0], orient = 'h'); #palette="Blues");
plt.show()


# <a id=section411></a>

# ### 4.11 Which is the best venue to defend the total?

# In[28]:


venue_by_runs=ipl[ipl['win_by_runs']>0].groupby(['venue'])['win_by_runs'].count().sort_values(ascending = False)
venue_by_runs


# In[29]:


plt.figure(figsize=(5,12))
sns.swarmplot(x='win_by_runs',y='venue',data=ipl)
plt.show()


# **Wankhede Stadium** and **Feroz Shah Kotla** are the best venue to defend the total. 

# <a id=section412></a>

# ### 4.12 Which is the best venue to chase the total?

# In[30]:


venue_by_wickets=ipl[ipl['win_by_wickets']>0].groupby(['venue'])['win_by_wickets'].count().sort_values(ascending = False)
venue_by_wickets


# In[31]:


plt.figure(figsize=(5,12))
sns.swarmplot(x='win_by_wickets',y='venue',data=ipl)
plt.show()


# **Eden Gardens** is the best venue for chasing

# <a id=section413></a>

# ### 4.13 Venue wise result!

# In[33]:


venue_details=pd.concat([venue_by_runs, venue_by_wickets], axis=1)
venue_details


# In[34]:


print(venue_details.dtypes)


# In[35]:


venue_details['win_by_runs'] = venue_details['win_by_runs'].fillna(0.0).astype(int) 
venue_details


# <a id=section414></a>

# ### 4.14 Which player received the maximum number of Player of Match award while defending the total?

# In[40]:


MOM_runs=ipl[ipl['win_by_runs']>2].groupby(['player_of_match'])['win_by_runs'].count().sort_values(ascending = False)
MOM_runs


# <a id=section415></a>

# ### 4.15 Which player received the maximum number of Player of Match award while chasing the target?

# In[41]:


MOM_wickets=ipl[ipl['win_by_wickets']>2].groupby(['player_of_match'])['win_by_wickets'].count().sort_values(ascending = False)
MOM_wickets


# <a id=section416></a>

# ### 4.16 Player of the Match - Match wise result

# In[42]:


MOM_details=pd.concat([MOM_runs, MOM_wickets], axis=1)
MOM_details


# In[43]:


print(MOM_details.dtypes)


# In[44]:


MOM_details['win_by_runs'] = MOM_details['win_by_runs'].fillna(0.0).astype(int) 
MOM_details


# In[92]:


pd.set_option('display.max_rows',1000)
MOM_details['win_by_wickets'] = MOM_details['win_by_wickets'].fillna(0.0).astype(int)
print(MOM_details)


# <a id=section417></a>

# ### Venue & Player of The Match wise results!

# In[78]:


MOM_Venue=ipl.loc[ : , 'player_of_match':'venue']
byVenue = MOM_Venue.groupby('venue')
byVenue.describe()


# <a id=section7></a>

# ### Conclusion

# ## Cricket one of the most loved and favourite sports entertainment specially in India. Here I present data analysis for IPL (Indian Premier League) matches from the beginning till year 2018. This can be useful for all the cricket lovers to analyse and made quick decisions based out from this. Few of which can be like; Which cricketer has scored the most for a particular season? or Which factors affect winning rate ? few of which can be toss decision and venue of the match played ! Various visual along with custom one have been used with filters and drill down capability. 
# ## Also it will addresses the player profiling system which can be a great help for the team leaders on the auction day. The statistics of 644 matches have been used in the experiments. 
