# -*- coding: utf-8 -*-
''' Code for HW1, Q1 - Kiana Mittelstaedt
    This runs (in full) in Google Colab
    Please put the file "soc-LiveJournal1Adj.txt" in your "Files"
    The output file "part-00000" saves in a folder called "finaloutputQ1"
    NOTE: this took a very long time to run (about 10 minutes) in my Google
    Colab once all was combined into one file. It did not take as long when
    it was broken up into different segments 
'''
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
import pandas as pd
import itertools 

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

# load the dataset 
file = sc.textFile("soc-LiveJournal1Adj.txt")

# map each user to all of his/her friends (user,[friends]), note some users don't have any friends
user_list = file.map(lambda user: (user.split('\t')[0], user.split('\t')[1].split(',') ))

# this creates all permutations of (friend,friend) pairs 
def friend_combos(line):
  user = line[0]
  friends= line[1]
  return [(val,1) for val in itertools.permutations(friends, 2)]

friend_tups = user_list.map(friend_combos).flatMap(lambda x: x)

# this creates all (user,friend) pairs 
def user_friend_combos(line):
  user = line[0]
  friend = line[1]
  return [((user,f),1) for f in friend]

user_friends = user_list.map(user_friend_combos).flatMap(lambda x:x)

# this removes any pairs that are already friends
remove_friends = friend_tups.subtract(user_friends)

# this is ((pair of users), # of mutual friends)
grouped_remove_friends = remove_friends.reduceByKey(lambda x,y: x+y)

# rearrange so it is (user ID, (# mutual friends, user))
restructured = grouped_remove_friends.map(lambda i : (i[0][0],(i[1],i[0][1])))

# this is grouped by userID (userID, [(# of mutual friends, user),...])
sort_restruct = restructured.groupByKey().map(lambda x : (x[0],list(x[1])))

# sorts recommended users in ascending order 
def sorting(some_line):
  user = some_line[0]
  list_of_tuples = some_line[1]
  sorted_tup_list = sorted(list_of_tuples, key=lambda x:int(x[1]))
  return [user, sorted_tup_list]

sorted_mutual_friends = sort_restruct.map(sorting)

# after sorting by user ID, now sort by number of mutual friends
# and take top 10 recommendations 
def recommendations(some_line):
  user = some_line[0]
  list_of_tuples = some_line[1]
  sorted_rec_list = sorted(list_of_tuples, key=lambda x:-x[0])
  return [user, sorted_rec_list[:10]]

recommended_friends = sorted_mutual_friends.map(recommendations)

# this returns just the userIDs and not the # of mutual friends 
def just_users(something):
  user = something[0]
  rec_list = something[1]
  user_ids = [name[1] for name in rec_list]
  return [user, user_ids]

user_recommendations = recommended_friends.map(just_users)

# final output 
# this saves to a folder called "finaloutputQ1" 
# the .txt file is called "part-00000 by default" 
final_output = user_recommendations.map(lambda x: "{}\t{}"
.format(x[0],",".join(map(str,x[1])))
).coalesce(1,True).saveAsTextFile('finaloutputQ1')
