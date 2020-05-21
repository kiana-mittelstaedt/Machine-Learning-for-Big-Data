# -*- coding: utf-8 -*-
''' Code for HW1, Q1 - Kiana Mittelstaedt
    This runs (in full) in Google Colab
    Please put the file "browsing.txt" in your "Files"
    The output file "part-00000" saves in a folder called "finaloutputQ2d"
    The result is in the form [('item A','item B'),confidence score], so
    the rule is conf(A -> B)
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
file = sc.textFile("browsing.txt")

# this is the first pass L_1 
# returns 647 frequent items 
def one_item(line):
  split_string = line.split(' ')
  ind_items = [s for s in split_string if s != '']
  return [(val,1) for val in ind_items]

each_item = file.map(one_item).flatMap(lambda x:x)
tmp_count_items = each_item.reduceByKey(lambda x,y: x+y)
pass1_items = tmp_count_items.filter(lambda x: x[1]>=100)

# this creates all combinations of (item,item) pairs 
def item_combos(line):
  split_string = sorted(line.split(' '))
  ind_items = [s for s in split_string if s != '']
  all_pairs = [(val,1) for val in itertools.combinations(ind_items, 2)]
  return all_pairs

# reduce by key 
# this gives [((item,item), # times they are searched in same session)]
tmp_freq_pairs = file.map(item_combos).flatMap(lambda x: x).reduceByKey(lambda x,y: x+y)

sorted_item_count1 = tmp_freq_pairs.filter(lambda x: x[1]>=100)

# rearrange the items in the pair to make assoc. rules easier
sorted_item_count2 = sorted_item_count1.map(lambda i : ( (i[0][1],i[0][0]) , i[1]))

# merge the frequent pair lists
freq_pairs = sorted_item_count1.union(sorted_item_count2)

# turn frequent items into a dictionary so it can be searched through
freq_items = pass1_items.collectAsMap()
keys = freq_items.keys()

def conf_calc(pairs): 
  id1 = pairs[0][0]
  id1_count = freq_items.get(id1)
  conf_score = pairs[1]/id1_count
  return [pairs[0],conf_score]

# how often the second one shows up given that the first one is there 
confidence_scores = freq_pairs.map(conf_calc).sortBy(lambda x: -x[1])

confidence_scores.coalesce(1,True).saveAsTextFile('finaloutputQ2d.txt')
