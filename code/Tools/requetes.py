
from pyspark.sql import SparkSession
from pyspark.sql.functions import when,concat,count, desc,asc, col, avg ,sum, regexp_extract

#convertir une dataframe en liste
def convert_tolist(df,column):
	return df.select(column).toPandas()[column].tolist()

#culculer le nombre de tuple par groupe
def count_groupBy(df, co,condition='1=1' ):
	return df.groupBy(co).count().where(condition)

#recuperer les premiers avec un tri desc
def top_line(df, ntop, co):
	return df.sort(desc(co)).limit(ntop)

#calculer la moyenne par groupes
def avg_groupBy(df, co, avrage,condition='1=1'):
	return df.where(condition).groupBy(co).agg(avg(col(avrage)))
