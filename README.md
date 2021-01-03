# Analyse-Text-Mining-data-set-E-commerce-Pyspark-

## Analyse des donnée

### Présentation de base de données

Il s’agit d’une base de données de E-commerce de vêtements pour femmes qui tourne autour des critiques
écrites par les clients. Il s’agit de véritables données commerciales.
- Lien vers données https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

## Details des variables

Clothing ID : (INT) Variable catégorielle qui représente l’article noté et donne sont avis.\\
Age : (INT) Age du client.
Title : (STRING) Le titre de l’avis.
Review Text : (STRING) Avis du client.
Rating : (INT) La note.
Recommended IND : (INT) Variable binaire 1 si le client recommande le produit sinon 0.
Positive Feedback Count : (INT) Nombre de personnes d’accord avec le commentaire.
Division Name : (STRING) Nom de la division du produit.
Department Name : (STRING) Nom du department du produit.
Class Name : (STRING) Nom de la class du produit.
Sentiment : (INT) C’est une colonne que j’ai crée. Si la note est moins de la moyenne ou égal 3 c’est un avis
négatif(-1) si elle est plus grande donc l’avis est positif(1).
