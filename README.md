# Analyse-Text-Mining-data-set-E-commerce-Pyspark-

## Analyse des donnée

### Présentation de base de données

Il s’agit d’une base de données de E-commerce de vêtements pour femmes qui tourne autour des critiques
écrites par les clients. Il s’agit de véritables données commerciales.
- Lien vers données https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

## Details des variables

- Clothing ID : (INT) Variable catégorielle qui représente l’article noté et donne sont avis.\\
- Age : (INT) Age du client.
- Title : (STRING) Le titre de l’avis.
- Review Text : (STRING) Avis du client.
- Rating : (INT) La note.
- Recommended IND : (INT) Variable binaire 1 si le client recommande le produit sinon 0.
- Positive Feedback Count : (INT) Nombre de personnes d’accord avec le commentaire.
- Division Name : (STRING) Nom de la division du produit.
- Department Name : (STRING) Nom du department du produit.
- Class Name : (STRING) Nom de la class du produit.
- Sentiment : (INT) C’est une colonne que j’ai crée. Si la note est moins de la moyenne ou égal 3 c’est un avis
- négatif(-1) si elle est plus grande donc l’avis est positif(1).

## Code 

### Problématique

Analyse de sentiments(’Sentiment’) et prédiction de Note si c’est une bonne note ou mauvaise grace au
commentaire (’Review Text’) en analysant le texte(Texte Mining).

### Framework installer

- Pyspark
- Pandas (pip3 install Pandas)
- Numpy (pip3 install )
- Matplotlib (pip3 install Numpy)
- Seaborn (pip3 install Seaborn)
- Wordcloud (pip3 install Wordcloud)

### Code source
- Dossier (Package) "Tools" : Contient les fichiers résponsable de notre traitement de données.
- Dossier "models" : C’est la ou on stock nos models de ML une fois leur apprentissage est fini.
- Dossier "plot" : Dossier de sauvgarder des figures Plot (Histo, Pie, Wordcloud...) en forme image (Png).
- Fichier "main.py" : Notre fichier principale contient les apelles de fonctions pour notre traitement et prrédiction.
- Fichier "analyseData.py" : Notre fichier ou on a nos fonctions d’analyse de text via Matplotlib.
- Fichier "prepareData.py" : Notre fichier La ou on a nos fonctions de préparation de données (Netoyage,
Enrichissement de données...).
- Fichier "funcMl.py" : Notre fichier résponsable du traitement de machine learning (prédiction, classification, évaluation...).
- Fichier "requetes.py" : Notre fichier contient une simplification de nos requetes sparkSQL.

### Run code

Pour compiler notre code il suffit de compiler notre fichier "python3 main.py", une fois notre code est éxécuté on on obtient les détail de chaque étape sur notre terminal :
- Etape1 : Lire notre fichier CSV est avoir les details sur notre DF 
- Etape2 : Traiter et Netoyer nos données est avoir les details sur notre DF Clean
- Etape3 : Analyser de nos données et les sauvgarder dans le dossier "plot".
- Etape4 : : Appliquer nos models de machine learning sur nos données, les sauvgarder dans le dossier "model"
retourner l’évaluation de chaque model utiliser (taux d’erreurs).
