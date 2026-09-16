#!/usr/bin/env python
# coding: utf-8

# ## modelisation
# 
# null

# # 04 - Modélisation Machine Learning
# 
# ## Détection des moteurs potentiellement défectueux
# 
# L'objectif de cette étape est de développer plusieurs modèles de
# classification permettant d'identifier les moteurs présentant une
# anomalie à partir des caractéristiques statistiques issues des tests.
# 
# Les modèles seront entraînés à partir de la table Gold
# `gold_fait_test_moteur`.
# 
# Plusieurs algorithmes seront comparés afin d'identifier celui qui offre
# le meilleur compromis entre détection des anomalies et limitation des
# fausses alertes.

# ## 1. Importation des bibliothèques
# 
# Les bibliothèques PySpark nécessaires à la préparation des données,
# à l'entraînement des modèles et à leur évaluation sont importées.

# In[4]:


from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    LinearSVC
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


# ## 2. Chargement des données Gold
# 
# La table `gold_fait_test_moteur` constitue le dataset final destiné
# à la modélisation.
# 
# Chaque ligne correspond à un moteur testé et contient :
# 
# - les caractéristiques statistiques calculées à partir des capteurs ;
# - l'identifiant du moteur ;
# - les informations associées au test ;
# - la variable cible `LABEL`.
# 
# La variable `LABEL` représente la présence ou non d'une anomalie.

# In[5]:


from pyspark.sql import functions as F

df = spark.sql("""
    SELECT *
    FROM gold_lakehouse.gold_fait_test_moteur
""")

print("Nombre de lignes :", df.count())
print("Nombre de colonnes :", len(df.columns))

display(df.limit(10))


# ## 3. Analyse de la variable cible
# 
# La variable `LABEL` est la variable cible du problème de classification.
# 
# - `LABEL = 0` : moteur considéré comme normal ;
# - `LABEL = 1` : moteur présentant une anomalie.
# 
# Une première analyse de la distribution des classes est réalisée afin
# d'identifier un éventuel déséquilibre des données.

# In[6]:


label_distribution = (
    df.groupBy("LABEL")
      .count()
      .orderBy("LABEL")
)

display(label_distribution)


# In[7]:


total = df.count()

display(
    label_distribution.withColumn(
        "percentage",
        F.round(F.col("count") / total * 100, 2)
    )
)


# ## 4. Vérification de l'unicité des moteurs
# 
# Le grain de la table Gold est défini au niveau du moteur.
# 
# Chaque `SERIAL_NUMBER` doit donc apparaître une seule fois dans le
# dataset utilisé pour la modélisation.

# In[8]:


total_rows = df.count()
unique_serials = df.select("SERIAL_NUMBER").distinct().count()

print("Nombre total de lignes :", total_rows)
print("Nombre de moteurs uniques :", unique_serials)

if total_rows == unique_serials:
    print("✓ Un moteur correspond bien à une seule ligne.")
else:
    print("⚠ Des doublons sont présents.")


# ## 5. Vérification des valeurs manquantes
# 
# Les valeurs manquantes peuvent empêcher ou perturber l'entraînement des
# modèles.
# 
# Un contrôle est donc effectué sur l'ensemble des colonnes.

# In[9]:


null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

display(null_counts)


# ## 6. Sélection des variables explicatives
# 
# Les variables utilisées pour l'apprentissage correspondent aux
# caractéristiques statistiques calculées à partir des capteurs.
# 
# Pour éviter toute fuite d'information, les éléments suivants sont
# exclus :
# 
# - `SERIAL_NUMBER` : identifiant du moteur 
# - `LABEL` : variable cible 
# - `PRODUCT_NUMBER` et `TEST_STAND` : informations catégorielles liées
#   au contexte du test 
# - les variables `_leak_*` : variables décrivant le déroulement du test
#   plutôt que l'état du moteur
# 
# Les 60 caractéristiques issues des capteurs sont ainsi utilisées comme
# variables explicatives.

# In[10]:


exclude_cols = [
    "SERIAL_NUMBER",
    "LABEL",
    "PRODUCT_NUMBER",
    "TEST_STAND",
    "_leak_nb_mesures",
    "_leak_nb_programmes",
    "_leak_duree_sec"
]

feature_cols = [
    c for c in df.columns
    if c not in exclude_cols
]

print("Nombre de variables explicatives :", len(feature_cols))
print(feature_cols)


# ## 7. Vérification des types des variables
# 
# Les variables explicatives doivent être numériques afin d'être utilisées
# par les algorithmes de Machine Learning de PySpark.

# In[11]:


from pyspark.sql.types import NumericType

non_numeric_cols = [
    field.name
    for field in df.select(feature_cols).schema.fields
    if not isinstance(field.dataType, NumericType)
]

if not non_numeric_cols:
    print("Toutes les variables sont numériques.")
else:
    print("Variables non numériques :", non_numeric_cols)


# ## 8. Analyse statistique des variables
# 
# Une analyse descriptive des variables explicatives est réalisée afin
# d'identifier leurs principales caractéristiques et d'éventuelles
# valeurs atypiques.

# In[12]:


display(
    df.select(feature_cols).describe()
)


# ## 9. Comparaison des caractéristiques selon le label
# 
# Afin d'identifier d'éventuelles différences entre les moteurs normaux
# et les moteurs présentant une anomalie, les moyennes de certaines
# caractéristiques sont comparées selon la valeur de `LABEL`.
# 
# Cette analyse permet également de mieux comprendre les variables
# susceptibles d'être discriminantes pour la modélisation.

# In[13]:


selected_features = [
    "PR_1_mean",
    "PR_1_std",
    "PR_1_min",
    "PR_1_max",
    "TEMP_1_mean",
    "TEMP_1_std",
    "SPEED_1_mean",
    "SPEED_1_max"
]

display(
    df.groupBy("LABEL")
      .agg(*[
          F.round(F.avg(c), 3).alias(c)
          for c in selected_features
      ])
)


# ## 10. Préparation du dataset pour le Machine Learning
# 
# Le dataset utilisé pour l'apprentissage est constitué uniquement de la
# variable cible `LABEL` et des 60 variables explicatives.
# 
# Les identifiants et informations exclues précédemment ne sont pas
# transmis aux modèles.

# In[14]:


ml_df = df.select(
    "SERIAL_NUMBER",
    "LABEL",
    *feature_cols
)

display(ml_df.limit(10))


# ## 11. Séparation des données d'entraînement et de test
# 
# Le dataset est séparé en deux ensembles :
# 
# - un ensemble d'entraînement utilisé pour construire les modèles ;
# - un ensemble de test utilisé uniquement pour évaluer leurs performances.
# 
# Le dataset étant fortement déséquilibré, la séparation est réalisée
# séparément pour les deux classes afin de conserver des observations
# anormales dans les ensembles d'entraînement et de test.
# 
# Aucune opération de rééquilibrage n'est réalisée avant cette séparation
# afin d'éviter une fuite de données.

# In[15]:


df_normal = ml_df.filter(F.col("LABEL") == 0)
df_anomaly = ml_df.filter(F.col("LABEL") == 1)

train_normal, test_normal = df_normal.randomSplit(
    [0.8, 0.2],
    seed=42
)

train_anomaly, test_anomaly = df_anomaly.randomSplit(
    [0.8, 0.2],
    seed=42
)

train_df = train_normal.union(train_anomaly)
test_df = test_normal.union(test_anomaly)

print("Train :", train_df.count())
print("Test  :", test_df.count())


# ## 12. Vérification de la répartition des classes
# 
# La répartition des classes est vérifiée après la séparation afin de
# s'assurer que les deux ensembles contiennent bien des moteurs normaux
# et des moteurs présentant une anomalie.

# In[16]:


print("TRAIN")
display(
    train_df.groupBy("LABEL")
            .count()
            .orderBy("LABEL")
)

print("TEST")
display(
    test_df.groupBy("LABEL")
           .count()
           .orderBy("LABEL")
)


# ## 13. Gestion du déséquilibre des classes
# 
# Le dataset présente un déséquilibre important entre les moteurs normaux
# et les moteurs présentant une anomalie.
# 
# Une pondération des classes est donc utilisée pendant l'apprentissage.
# 
# Les observations de la classe minoritaire (`LABEL = 1`) reçoivent un poids
# plus important afin d'éviter que les modèles privilégient uniquement la
# classe majoritaire.
# 
# Les poids sont calculés uniquement sur les données d'entraînement.

# In[17]:


class_counts = (
    train_df.groupBy("LABEL")
            .count()
            .collect()
)

counts = {
    int(row["LABEL"]): row["count"]
    for row in class_counts
}

n0 = counts.get(0, 0)
n1 = counts.get(1, 0)

total_train = n0 + n1

weight_0 = total_train / (2 * n0)
weight_1 = total_train / (2 * n1)

print("Nombre de normaux :", n0)
print("Nombre d'anomalies :", n1)

print("Poids classe 0 :", weight_0)
print("Poids classe 1 :", weight_1)


# In[18]:


train_df = train_df.withColumn(
    "classWeight",
    F.when(F.col("LABEL") == 1, F.lit(weight_1))
     .otherwise(F.lit(weight_0))
)

display(
    train_df.groupBy("LABEL")
            .agg(
                F.count("*").alias("count"),
                F.first("classWeight").alias("weight")
            )
)


# ## 14. Transformation des variables en vecteur
# 
# Les algorithmes de Machine Learning de Spark ML utilisent une colonne
# `features` contenant un vecteur numérique.
# 
# Les 60 variables explicatives sont donc regroupées à l'aide de
# `VectorAssembler`.

# In[19]:


assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)


# # 15. Modèle 1 - Régression Logistique
# 
# La régression logistique constitue le modèle de référence de cette
# étude.
# 
# Elle permet de modéliser la probabilité qu'un moteur appartienne à la
# classe des moteurs présentant une anomalie.
# 
# Son principal avantage est son caractère interprétable, ce qui permet
# d'analyser l'influence des variables utilisées.
# 
# $$P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p)}}$$
# 
# où :
# 
# - $P(y=1 \mid x)$ : probabilité que le moteur soit défectueux.
# - $x_1, x_2, \dots, x_p$ : les features en entrée (les 60 statistiques capteurs).
# - $\beta_0$ : la valeur de base du modèle (biais).
# - $\beta_1, \beta_2, \dots, \beta_p$ : le poids de chaque feature.
# - $e$ : la constante d'Euler, qui ramène le résultat entre 0 et 1.
# 
# 

# In[20]:


lr = LogisticRegression(
    featuresCol="features",
    labelCol="LABEL",
    weightCol="classWeight",
    maxIter=100
)

lr_pipeline = Pipeline(
    stages=[
        assembler,
        lr
    ]
)

lr_model = lr_pipeline.fit(train_df)

lr_predictions = lr_model.transform(test_df)

display(
    lr_predictions.select(
        "SERIAL_NUMBER",
        "LABEL",
        "prediction",
        "probability"
    )
)


# # 16. Modèle 2 - Random Forest
# 
# Le Random Forest est un modèle basé sur un ensemble d'arbres de
# décision.
# 
# Il permet notamment de prendre en compte des relations non linéaires
# entre les caractéristiques des capteurs et la présence d'une anomalie.
# 
# Ce modèle est particulièrement intéressant pour les données tabulaires
# issues des tests moteurs.

# In[21]:


rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="LABEL",
    weightCol="classWeight",
    numTrees=200,
    maxDepth=8,
    seed=42
)

rf_pipeline = Pipeline(
    stages=[
        assembler,
        rf
    ]
)

rf_model = rf_pipeline.fit(train_df)

rf_predictions = rf_model.transform(test_df)

display(
    rf_predictions.select(
        "SERIAL_NUMBER",
        "LABEL",
        "prediction",
        "probability"
    )
)


# # 17. Modèle 3 - Gradient-Boosted Trees
# 
# Le Gradient-Boosted Trees (GBT) repose sur une succession d'arbres de
# décision.
# 
# Chaque nouvel arbre cherche à corriger les erreurs produites par les
# arbres précédents.
# 
# Ce modèle permet de capturer des relations complexes et non linéaires
# entre les variables issues des capteurs.

# In[19]:


gbt = GBTClassifier(
    featuresCol="features",
    labelCol="LABEL",
    weightCol="classWeight",
    maxIter=50,
    maxDepth=5,
    seed=42
)

gbt_pipeline = Pipeline(
    stages=[
        assembler,
        gbt
    ]
)

gbt_model = gbt_pipeline.fit(train_df)

gbt_predictions = gbt_model.transform(test_df)

display(
    gbt_predictions.select(
        "SERIAL_NUMBER",
        "LABEL",
        "prediction",
        "probability"
    )
)


# # 18. Modèle 4 - Linear SVM
# 
# Le Linear Support Vector Machine (SVM) est un modèle de classification
# qui cherche à déterminer une frontière séparant les deux classes.
# 
# Il est utilisé ici comme modèle complémentaire à la régression
# logistique afin de comparer deux approches linéaires différentes.

# In[20]:


svm = LinearSVC(
    featuresCol="features",
    labelCol="LABEL",
    weightCol="classWeight",
    maxIter=100
)

svm_pipeline = Pipeline(
    stages=[
        assembler,
        svm
    ]
)

svm_model = svm_pipeline.fit(train_df)

svm_predictions = svm_model.transform(test_df)

display(
    svm_predictions.select(
        "SERIAL_NUMBER",
        "LABEL",
        "prediction"
    )
)


# # 19. Évaluation des modèles
# 
# La performance des modèles est évaluée à partir de plusieurs métriques.
# 
# L'accuracy seule n'est pas suffisante dans notre cas en raison du fort
# déséquilibre entre les deux classes.
# 
# Les métriques principales sont :
# 
# - **Precision** : proportion des moteurs détectés comme anormaux qui
#   sont réellement anormaux ;
# - **Recall** : proportion des moteurs réellement anormaux correctement
#   détectés ;
# - **F1-score** : compromis entre Precision et Recall ;
# - **ROC-AUC** : capacité globale du modèle à distinguer les deux classes.
# 
# Dans le contexte industriel étudié, le **Recall de la classe anomalie**
# est particulièrement important car manquer un moteur potentiellement
# défectueux peut avoir des conséquences lors de son installation chez
# le client.

# # 20. Fonction d'évaluation

# In[21]:


def evaluate_model(predictions, model_name):
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="LABEL",
        predictionCol="prediction",
        metricName="f1"
    )
    
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="LABEL",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    f1 = evaluator_f1.evaluate(predictions)
    accuracy = evaluator_accuracy.evaluate(predictions)
    
    # Matrice de confusion
    cm = (
        predictions.groupBy("LABEL", "prediction")
                   .count()
                   .orderBy("LABEL", "prediction")
    )
    
    print("=" * 50)
    print(model_name)
    print("=" * 50)
    print("Accuracy :", round(accuracy, 4))
    print("F1-score :", round(f1, 4))
    
    display(cm)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1": f1
    }


# # 21. Évaluer les 4 modèles

# In[22]:


results_lr = evaluate_model(
    lr_predictions,
    "Logistic Regression"
)

results_rf = evaluate_model(
    rf_predictions,
    "Random Forest"
)

results_gbt = evaluate_model(
    gbt_predictions,
    "GBT"
)

results_svm = evaluate_model(
    svm_predictions,
    "Linear SVM"
)


# ## 22. Precision et Recall de la classe anomalie
# 
# La matrice de confusion permet d'identifier les différents types de
# prédictions :
# 
# - **True Positive (TP)** : anomalie correctement détectée ;
# - **False Positive (FP)** : moteur normal considéré à tort comme
#   anormal ;
# - **True Negative (TN)** : moteur normal correctement identifié ;
# - **False Negative (FN)** : anomalie non détectée.
# 
# Le Recall de la classe 1 est particulièrement important dans ce projet.

# In[23]:


def classification_metrics(predictions, model_name):
    
    cm = (
        predictions.groupBy("LABEL", "prediction")
                   .count()
                   .collect()
    )
    
    values = {
        (int(row["LABEL"]), int(row["prediction"])): row["count"]
        for row in cm
    }
    
    TP = values.get((1, 1), 0)
    FN = values.get((1, 0), 0)
    FP = values.get((0, 1), 0)
    TN = values.get((0, 0), 0)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    
    return (
        model_name,
        TP,
        FP,
        TN,
        FN,
        precision,
        recall,
        f1
    )

    


# In[24]:


all_metrics = [
    classification_metrics(lr_predictions, "Logistic Regression"),
    classification_metrics(rf_predictions, "Random Forest"),
    classification_metrics(gbt_predictions, "GBT"),
    classification_metrics(svm_predictions, "Linear SVM")
]

columns = ["Model", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1"]

# Normalisation explicite des types avant de construire le DataFrame Spark :
# TP/FP/TN/FN toujours en int, Precision/Recall/F1 toujours en float.
# Ça évite le CANNOT_MERGE_TYPE (Spark refusait de fusionner LongType et
# DoubleType détectés sur des lignes différentes pour la même colonne).
all_metrics_clean = [
    (
        str(row[0]),
        int(row[1]), int(row[2]), int(row[3]), int(row[4]),
        float(row[5]), float(row[6]), float(row[7]),
    )
    for row in all_metrics
]

metrics_df = spark.createDataFrame(all_metrics_clean, columns)

display(metrics_df.orderBy(F.desc("F1")))


# ## 24. Évaluation avec ROC-AUC
# 
# La métrique ROC-AUC mesure la capacité du modèle à distinguer les
# moteurs normaux des moteurs présentant une anomalie.
# 
# Une valeur proche de 1 indique une bonne capacité de discrimination,
# tandis qu'une valeur proche de 0,5 correspond à une performance proche
# d'une classification aléatoire.

# In[25]:


evaluator_roc = BinaryClassificationEvaluator(
    labelCol="LABEL",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

roc_results = []

for name, predictions in [
    ("Logistic Regression", lr_predictions),
    ("Random Forest", rf_predictions),
    ("GBT", gbt_predictions),
    ("Linear SVM", svm_predictions)
]:
    
    auc = evaluator_roc.evaluate(predictions)
    
    roc_results.append(
        (name, auc)
    )

roc_df = spark.createDataFrame(
    roc_results,
    ["Model", "ROC_AUC"]
)

display(
    roc_df.orderBy(F.desc("ROC_AUC"))
)


# ## 25. Importance des variables
# 
# Les modèles Random Forest et GBT permettent d'estimer l'importance des
# variables utilisées dans la classification.
# 
# Cette analyse permet d'identifier les caractéristiques des capteurs
# qui contribuent le plus à la détection des anomalies.
# 
# Elle peut également apporter une première interprétation métier des
# résultats du modèle.

# In[26]:


from pyspark.sql.types import StructType, StructField, StringType, DoubleType

rf_algorithm = rf_model.stages[-1]
importances_array = rf_algorithm.featureImportances.toArray()

# Vérification de sécurité : si ces deux longueurs ne correspondent pas,
# le zip() tronquerait silencieusement sur la plus courte des deux listes.
assert len(feature_cols) == len(importances_array), (
    f"Mismatch : {len(feature_cols)} features vs {len(importances_array)} "
    f"importances — vérifier que feature_cols correspond bien aux colonnes "
    f"utilisées par le VectorAssembler de ce modèle."
)

# .tolist() convertit chaque numpy.float64 en float Python natif : c'est ce
# qui manquait pour que Spark puisse déduire le type de la colonne Importance.
feature_importance_rf = list(zip(feature_cols, importances_array.tolist()))

# Schéma explicite : plus robuste, on ne dépend plus de l'inférence de Spark.
schema = StructType([
    StructField("Feature", StringType()),
    StructField("Importance", DoubleType()),
])

importance_rf_df = spark.createDataFrame(feature_importance_rf, schema)

display(
    importance_rf_df.orderBy(F.desc("Importance")).limit(20)
)


# In[27]:


gbt_algorithm = gbt_model.stages[-1]
importances_array = gbt_algorithm.featureImportances.toArray()

assert len(feature_cols) == len(importances_array), (
    f"Mismatch : {len(feature_cols)} features vs {len(importances_array)} "
    f"importances — vérifier que feature_cols correspond bien aux colonnes "
    f"utilisées par le VectorAssembler de ce modèle."
)

feature_importance_gbt = list(zip(feature_cols, importances_array.tolist()))

schema = StructType([
    StructField("Feature", StringType()),
    StructField("Importance", DoubleType()),
])

importance_gbt_df = spark.createDataFrame(feature_importance_gbt, schema)

display(
    importance_gbt_df.orderBy(F.desc("Importance")).limit(20)
)


# In[31]:


from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

# ============================================================
# Analyse des différents seuils - Random Forest
# ============================================================

# Conversion du vecteur probability en tableau
rf_threshold_df = (
    rf_predictions
    .withColumn(
        "prob_anomaly",
        vector_to_array("probability")[1]
    )
)

# Liste des seuils à tester
thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05
]

threshold_results = []

for threshold in thresholds:

    pred = (
        rf_threshold_df
        .withColumn(
            "prediction_threshold",
            (F.col("prob_anomaly") >= F.lit(threshold)).cast("int")
        )
    )

    TP = pred.filter(
        (F.col("LABEL") == 1) &
        (F.col("prediction_threshold") == 1)
    ).count()

    FN = pred.filter(
        (F.col("LABEL") == 1) &
        (F.col("prediction_threshold") == 0)
    ).count()

    FP = pred.filter(
        (F.col("LABEL") == 0) &
        (F.col("prediction_threshold") == 1)
    ).count()

    TN = pred.filter(
        (F.col("LABEL") == 0) &
        (F.col("prediction_threshold") == 0)
    ).count()

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    threshold_results.append(
        (
            float(threshold),
            int(TP),
            int(FP),
            int(TN),
            int(FN),
            float(precision),
            float(recall),
            float(f1)
        )
    )

# Création du DataFrame de comparaison
threshold_df = spark.createDataFrame(
    threshold_results,
    [
        "Threshold",
        "TP",
        "FP",
        "TN",
        "FN",
        "Precision",
        "Recall",
        "F1"
    ]
)

# Affichage des résultats
display(
    threshold_df.orderBy(F.desc("Threshold"))
)


# In[32]:


evaluator_pr = BinaryClassificationEvaluator(
    labelCol="LABEL",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

for name, predictions in [
    ("Logistic Regression", lr_predictions),
    ("Random Forest", rf_predictions),
    ("GBT", gbt_predictions),
    ("Linear SVM", svm_predictions)
]:

    pr_auc = evaluator_pr.evaluate(predictions)

    print(
        name,
        "PR-AUC =",
        round(pr_auc, 4)
    )


# # 27. Sélection du modèle final
# 
# Les différents modèles sont comparés à partir de leurs performances.
# 
# Le choix du modèle final ne repose pas uniquement sur l'Accuracy.
# 
# Une attention particulière est portée au Recall de la classe `1`,
# correspondant aux moteurs présentant une anomalie.
# 
# Le modèle retenu doit permettre de détecter un maximum d'anomalies tout
# en conservant un niveau acceptable de fausses alertes.
# 
# La décision finale prend donc en compte :
# 
# - le Recall ;
# - la Precision ;
# - le F1-score ;
# - le ROC-AUC ;
# - la matrice de confusion ;
# - l'interprétabilité du modèle.

# ## 27. Sélection du modèle final
# 
# L'objectif de cette étape est de sélectionner le modèle le plus adapté à la détection des moteurs présentant une anomalie.
# 
# Le jeu de données étant fortement déséquilibré, avec seulement 25 moteurs anormaux sur 1 262 moteurs (1,98 %), l'Accuracy ne constitue pas à elle seule un indicateur suffisant pour comparer les modèles. Un modèle prédisant majoritairement la classe normale peut en effet obtenir une Accuracy élevée tout en détectant très peu d'anomalies.
# 
# Les performances obtenues sur le jeu de test sont synthétisées ci-dessous :
# 
# | Modèle | Accuracy | Recall anomalie | Precision anomalie | F1 anomalie | ROC-AUC | PR-AUC |
# |---|---:|---:|---:|---:|---:|---:|
# | Logistic Regression | 0,9697 | 0,1667 | 0,3333 | 0,2222 | 0,6044 | 0,2075 |
# | Random Forest | 0,9740 | 0,0000 | 0,0000 | 0,0000 | 0,8552 | 0,3313 |
# | GBT | 0,9784 | 0,1667 | 1,0000 | 0,2857 | 0,6441 | 0,3262 |
# | Linear SVM | 0,9437 | 0,3333 | 0,1818 | 0,2353 | 0,6148 | 0,0937 |
# 
# Les résultats montrent que les valeurs élevées d'Accuracy ne reflètent pas nécessairement la capacité des modèles à détecter les anomalies. Par exemple, le Random Forest obtient une Accuracy de 97,40 %, mais ne détecte aucune des six anomalies du jeu de test lorsque le seuil de décision standard est utilisé.
# 
# Le Linear SVM présente initialement le meilleur Recall avec 33,33 %, mais cette amélioration s'accompagne de neuf faux positifs et d'une faible Precision de 18,18 %. La Logistic Regression détecte une seule anomalie sur six et présente également des performances limitées en ROC-AUC et PR-AUC.
# 
# Le GBT obtient la meilleure Accuracy (97,84 %) et ne génère aucun faux positif. Cependant, il ne détecte qu'une anomalie sur les six présentes dans le jeu de test, soit un Recall de 16,67 %.
# 
# Le Random Forest présente un comportement différent. Malgré un Recall nul avec le seuil standard, il obtient le meilleur ROC-AUC (0,8552) ainsi que la meilleure PR-AUC (0,3313) parmi les modèles étudiés. Ces résultats indiquent qu'il possède la meilleure capacité globale à différencier les moteurs normaux des moteurs potentiellement anormaux, mais que le seuil de classification standard de 0,5 n'est pas adapté au fort déséquilibre des classes.
# 
# Une analyse complémentaire du seuil de décision a donc été réalisée. Avec un seuil fixé à 0,05, le Random Forest détecte 3 anomalies sur les 6 présentes dans le jeu de test. Le Recall atteint ainsi 50 %, avec 11 faux positifs, une Precision de 21,43 % et un F1-score de 30 %.
# 
# Le Random Forest est donc retenu comme modèle final dans le cadre de cette expérimentation. Ce choix repose principalement sur ses performances en ROC-AUC et PR-AUC, ainsi que sur sa capacité à améliorer significativement la détection des anomalies lorsque le seuil de décision est adapté.
# 
# Le seuil de 0,05 est utilisé dans la suite afin de privilégier la détection des moteurs potentiellement défectueux. Ce choix implique davantage de fausses alertes, mais permet de réduire le nombre d'anomalies non détectées, ce qui est cohérent avec l'objectif du projet.
# 
# Il convient néanmoins de rester prudent dans l'interprétation de ces résultats, car le jeu de test ne contient que six moteurs anormaux. Une validation sur un volume plus important de données anormales permettrait de confirmer la stabilité des performances observées.
# 

# ## 28. Prédictions avec le modèle final
# 
# Après comparaison des différents algorithmes, le Random Forest est retenu comme modèle final.
# 
# Le seuil de classification standard de 0,5 étant inadapté au déséquilibre des classes, un seuil de 0,05 est appliqué à la probabilité d'appartenance à la classe anomalie.
# 
# Pour chaque moteur, le modèle fournit :
# 
# - la probabilité estimée d'anomalie ;
# - la prédiction finale (`0` : moteur normal, `1` : anomalie potentielle) ;
# - lorsque le label réel est disponible, le résultat de la classification (vrai positif, faux positif, vrai négatif ou faux négatif).

# In[22]:


from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

# ============================================================
# 28. Prédictions finales avec le Random Forest
# ============================================================

SEUIL_FINAL = 0.05

# Application du modèle Random Forest au jeu de test
predictions_finales = (
    rf_model
    .transform(test_df)
    .withColumn(
        "PROBABILITE_ANOMALIE",
        vector_to_array("probability")[1]
    )
    .withColumn(
        "PREDICTION_FINALE",
        (F.col("PROBABILITE_ANOMALIE") >= SEUIL_FINAL).cast("int")
    )
)

display(
    predictions_finales.select(
        "SERIAL_NUMBER",
        "LABEL",
        "PROBABILITE_ANOMALIE",
        "PREDICTION_FINALE"
    ).orderBy(
        F.desc("PROBABILITE_ANOMALIE")
    )
)


# In[24]:


from pyspark.sql import functions as F

predictions_finales = predictions_finales.withColumn(
    "RESULTAT",
    F.when(
        (F.col("LABEL") == 1) & (F.col("PREDICTION_FINALE") == 1),
        "Vrai positif"
    )
    .when(
        (F.col("LABEL") == 0) & (F.col("PREDICTION_FINALE") == 1),
        "Faux positif"
    )
    .when(
        (F.col("LABEL") == 1) & (F.col("PREDICTION_FINALE") == 0),
        "Faux négatif"
    )
    .otherwise("Vrai négatif")
)

display(
    predictions_finales.select(
        "SERIAL_NUMBER",
        "LABEL",
        "PROBABILITE_ANOMALIE",
        "PREDICTION_FINALE",
        "RESULTAT"
    ).orderBy(F.desc("PROBABILITE_ANOMALIE"))
)


# In[25]:


display(
    predictions_finales
    .groupBy("RESULTAT")
    .count()
    .orderBy(F.desc("count"))
)


# In[26]:


predictions_finales.select(
    "SERIAL_NUMBER",
    "LABEL",
    "PROBABILITE_ANOMALIE",
    "PREDICTION_FINALE",
    "RESULTAT"
).write \
 .format("delta") \
 .mode("overwrite") \
 .saveAsTable("gold_lakehouse.predictions_moteurs")


# ### Analyse des prédictions finales
# 
# L'application du Random Forest avec un seuil de décision fixé à 0,05 permet de détecter 3 des 6 moteurs réellement anormaux présents dans le jeu de test, soit un Recall de 50 %.
# 
# Le modèle génère également 11 fausses alertes : ces moteurs sont identifiés comme potentiellement anormaux alors que leur label réel est normal. En contrepartie, 214 moteurs normaux sont correctement classés et 3 anomalies ne sont pas détectées.
# 
# Le choix d'un seuil inférieur au seuil standard de 0,5 permet donc d'améliorer la détection des anomalies au prix d'une augmentation du nombre de faux positifs. Dans le contexte du projet, ce compromis permet de privilégier la détection des moteurs potentiellement défectueux.
# 
# Les probabilités d'anomalie peuvent également être utilisées pour classer les moteurs selon leur niveau de risque et ainsi prioriser les contrôles complémentaires.
