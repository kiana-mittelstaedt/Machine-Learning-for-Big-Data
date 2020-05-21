# -*- coding: utf-8 -*-
''' Code for HW1, Q1 - Kiana Mittelstaedt
    This runs (in full) in Google Colab
    Please put the file "browsing.txt" in your "Files"
    The output file "part-00000" saves in a folder called "finaloutputQ2e"
    The result is in the form [('item A','item B'), 'item c', confidence score], so
    the rule is conf( (A,B) -> C)
    This takes about 5 min to run
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

sorted_item_count2 = sorted_item_count1.map(lambda i : ( (i[0][1],i[0][0]) , i[1]))
freq_pairs = sorted_item_count1.union(sorted_item_count2)

# this creates all combinations of (item,item,item) triples
def item_triples(line):
  split_string = sorted(line.split(' '))
  ind_items = [s for s in split_string if s != '']
  all_triples = [(val,1) for val in itertools.combinations(ind_items, 3)]
  return all_triples

# reduce by key 
# this gives [((item,item,item), # times they are searched in same session)]
tmp_freq_triples = file.map(item_triples).flatMap(lambda x: x).reduceByKey(lambda x,y: x+y)
triples_count1 = tmp_freq_triples.filter(lambda x: x[1]>=100)

# reorder so it makes reading the association rules easier 
rule1 = triples_count1.map(lambda i : ( (i[0][0],i[0][1]) , i[0][2], i[1]))
rule2 = triples_count1.map(lambda i : ( (i[0][0],i[0][2]) , i[0][1], i[1]))
rule3 = triples_count1.map(lambda i : ( (i[0][1],i[0][2]) , i[0][0], i[1]))

freq_triples = sc.union([rule1, rule2, rule3])

# now the keys in the dictionary are tuples, i.e., the pairs 
freq_pairs_dict = freq_pairs.collectAsMap()
keys = freq_pairs_dict.keys()
values = freq_pairs_dict.values()

# calculate the confidence 
def conf_calc(pairs): 
  id1 = pairs[0][0]
  id2 = pairs[0][1]
  id_count = freq_pairs_dict.get((id1,id2))
  conf_score = pairs[2]/id_count
  return [pairs[0],pairs[1],conf_score]

# how often the second one shows up given that the first pair is there 
confidence_scores = freq_triples.map(conf_calc).sortBy(lambda x: -x[2])

confidence_scores.coalesce(1,True).saveAsTextFile('finaloutputQ2e.txt')
