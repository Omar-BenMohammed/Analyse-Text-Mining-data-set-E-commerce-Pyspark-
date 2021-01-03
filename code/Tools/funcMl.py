from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes,LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, CountVectorizer,StopWordsRemover,RegexTokenizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Modeliser & Split datafram

def prep_data(df,text,label):

	df_final =  df.select(text,label)

	df_final = (df_final
	   .withColumnRenamed(text,'text'))

	# regular expression tokenizer
	regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
	# stop words
	add_stopwords = ["http","https","amp","rt","t","c","the"] 
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
	# bag of words count
	countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

	label_stringIdx = StringIndexer(inputCol = label, outputCol = "label")
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])

	# Fit the pipeline to training documents.
	pipelineFit = pipeline.fit(df_final)
	dataset = pipelineFit.transform(df_final)

	return dataset.randomSplit([0.7, 0.3], seed = 100)

#####################################################################################

#prédir avec naiveBayes
def naiveBayes_predict(trainingData,testData):
	
	print('\n************************ Apprentissage du  NaiveBayes ************************\n')

	nb = NaiveBayes()
	nbModel = nb.fit(trainingData)

	print('\n**************************** Sauvegarder le model ****************************\n')

	nbModel.save("./models/myNaiveBayesModel")
	
	return nbModel.transform(testData)

#####################################################################################

#prédir avec LogisticRegression
def logisticRegression_predict(trainingData,testData):
	
	print('\n************************ Apprentissage du  LogisticRegression ************************\n')

	lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
	lrModel = lr.fit(trainingData)

	print('\n******************************* Sauvegarder le model *********************************\n')

	lrModel.save("./models/myLogisticRegressionModel")

	return lrModel.transform(testData)



#####################################################################################

#prédir avec LogisticRegression
def randomForestClassifier_predict(trainingData,testData):

	print('\n************************ Apprentissage du  RandomForestClassifier ************************\n')
	
	rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = 100, maxDepth = 4, maxBins = 32)
	rfModel = rf.fit(trainingData)

	print('\n********************************** Sauvegarder le model **********************************\n')

	rfModel.save("./models/myRandomForestClassifierModel")

	return rfModel.transform(testData)


#####################################################################################
#retour pourcentage d'erreurs

def taux_err(predictions):
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label", predictionCol="prediction", metricName="accuracy")
	print("Le taux d'erreurs est de: ",int((1 - evaluator.evaluate(predictions))*100), "%")
	print('\n ################################################################################################################# \n')


