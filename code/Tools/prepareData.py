import pyspark 
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType , IntegerType , FloatType

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonction local 
from Tools.requetes import *



#####################################################################################################
#Fonctions utils

#print une liste 
def printListe(l):
  print('********************Colonne DataFrame*******************')
  for x in l:
    print('Colonne : ',x[0] ,'(',x[1],')')

  print('********************************************************')


def condition(r):
    if (0< int(r) < 4):
        label = -1
    elif(4 <= int(r) <6):
        label = 1
    else:
        label = 0
    return label



######################################################################################################

#Lire le fichier CSV et retourne dataframe 
def getDataframe():

	spark = SparkSession.builder \
	.master("local") \
	.appName("Project") \
	.getOrCreate()

	data_df = spark.read.csv("./data/data.csv",header=True,sep=",")

	data_df = data_df.select("*").where("Rating <= 5")


	return data_df


#Info Dataframe

def infoDf(df):

  print('\n')
  print('############################################# INFO DataFrame ######################################################')
  print('\n')
  print('\n')

  print('Nombre de ligne dans DF : ',df.count(), '\n')

  #Calcule valeurs NULL df  
  data_df = df.na.drop()
  printListe(data_df.dtypes)
  print((df.count()-df.na.drop().count())/float(df.count()),"% ","de valeur manquantes")
  print('\n')
  df.show(5)




#Netoyage du datarame et crée columns utils
def cleanDf(data):


	data_df = data['Age','Review Text','Rating','Class Name','sentiment']
	print(data_df.count())
	data_df = data_df.na.drop()

	data_df = data_df.withColumn("Age", data_df["Age"].cast(IntegerType()))
	data_df = data_df.withColumn("Rating", data_df["Rating"].cast(IntegerType()))
	data_df = data_df.withColumn("sentiment", data_df["sentiment"].cast(IntegerType()))

	return data_df
