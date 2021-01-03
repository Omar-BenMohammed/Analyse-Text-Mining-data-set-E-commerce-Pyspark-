import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud


from Tools.requetes import *


def analyse_data_plot(data_df):

	plt.subplots(figsize=(14,6))

	wordcloud = WordCloud(background_color='white',width=900, height=300 ).generate(" ".join(data_df.toPandas()['Review Text']))

	plt.imshow(wordcloud)
	plt.title('Mots plus frequents des Revue\n \n',size=20)
	plt.axis('off')
	plt.savefig("./plot/wordcloud_Review_Text.png",bbox_inches = "tight")

	###################################################################################################################################

	plt.subplots(figsize=(14,6))

	wordcloud = WordCloud(background_color='white',width=900, height=300 ).generate(" ".join(data_df.toPandas()['Class Name']))

	plt.imshow(wordcloud)
	plt.title('Classe plus frequentes \n',size=20)
	plt.axis('off')
	plt.savefig("./plot/wordcloud_Class_Name.png",bbox_inches = "tight")

	###################################################################################################################################

	df_positive = data_df.select('*').where(" Rating >= 4")
	df_negative = data_df.select('*').where(" Rating < 4")


	plt.figure(figsize=(10,8))
	plt.title('Sentiment vs Age\n',size=20)
	sns.distplot(convert_tolist(df_positive,'Age'),label='Positive', hist=False)
	sns.distplot(convert_tolist(df_negative,'Age'),label='Positive', hist=False)

	plt.savefig("./plot/Sentiment_vs_Age.png",bbox_inches = "tight")

	###################################################################################################################################

	count_rating = count_groupBy(data_df, 'Rating')


	list_Rating = convert_tolist(count_rating, "Rating")
	list_count_Rating = convert_tolist(count_rating, "count")
	list_Rating[0]=0
	plt.clf()
	labels = list_Rating
	reviews = list_count_Rating

	fig, ax = plt.subplots(figsize=(13,10))
	w,a,b = ax.pie(reviews, autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})
	plt.title("Count Notes\n", size=20 )
	ax.legend(w, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

	plt.savefig("./plot/Count_Notes.png",bbox_inches = "tight")

	###################################################################################################################################

	count_sentiment = count_groupBy(data_df, 'sentiment')


	list_sentiment = convert_tolist(count_sentiment, "sentiment")
	list_count_sentiment = convert_tolist(count_sentiment, "count")
	plt.clf()
	labels = list_sentiment
	reviews = list_count_sentiment

	fig, ax = plt.subplots(figsize=(13,10))
	w,a,b = ax.pie(reviews, autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})
	plt.title("Count Sentiment\n", size=20 )
	ax.legend(w, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

	plt.savefig("./plot/Count_Sentiment.png",bbox_inches = "tight")

	###################################################################################################################################

	A = data_df.toPandas().groupby(['Class Name','sentiment'])['Class Name'].count().unstack('sentiment')
	A.plot(kind='bar', stacked=True,figsize=(13,10))
	plt.title("Sentiment vs Class",size=20)

	plt.savefig("./plot/Sentiment_vs_Class.png",bbox_inches = "tight")

	###################################################################################################################################

	corr = data_df.toPandas().corr()
	plt.subplots( figsize=(14, 7), sharex=True)
	ax = sns.heatmap(
	corr, 
	vmin=-1, vmax=1, center=0,
	cmap= sns.diverging_palette(20, 220, n=200),
	square=True
	)

	plt.savefig("./plot/Matrice_Corr",bbox_inches = "tight")