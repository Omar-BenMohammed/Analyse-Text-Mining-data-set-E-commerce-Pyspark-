# Fonction local 
from Tools.prepareData  import *
from Tools.funcMl import *
from Tools.analyseData import *




######################################################################################
#																					 #
# 							Traitements des donnees									 #
#																					 #
######################################################################################	

#Lire fichier CSV
file = getDataframe()
data = file

# Afficher les info de dataframe  
infoDf(data)

# Ajouter varible Sentiment
quality_udf = udf(lambda x: condition(x), StringType())
data = data.withColumn("sentiment", quality_udf("Rating"))

# Netoyer les données 
data_df = cleanDf(data)

# Afficher les info de dataframe
infoDf(data_df)


######################################################################################
#																					 #
#							Analyse des données via plot 							 #
#																					 #
######################################################################################	

analyse_data_plot(data_df)


######################################################################################
#																					 #
#								Machine Learning							 		 #
#																					 #
######################################################################################	

# Split data 
(trainingData, testData) = prep_data(data_df,'Review Text','sentiment')

# LogisticRegression Model
predictions_logisticRegression = logisticRegression_predict(trainingData,testData)
taux_err(predictions_logisticRegression)


# NaiveBayes Model
predictions_naiveBayes = naiveBayes_predict(trainingData,testData)
taux_err(predictions_naiveBayes)


# RandomForestClassifier Model
predictions_randomForestClassifier = randomForestClassifier_predict(trainingData,testData)
taux_err(predictions_randomForestClassifier)
