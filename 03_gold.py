#  MÉDAILLON  COUCHE GOLD : schéma sémantique (étoile) + dataset ML

#  Rôle : produire les tables prêtes pour l'analyse et le ML, organisées en
#         SCHÉMA ÉTOILE, avec le FEATURE ENGINEERING.

#  MODÉLISATION EN ÉTOILE :
#    - TABLE DE FAITS  (fait_test_moteur) : une ligne = un moteur, contient la
#      colonne CIBLE (LABEL) + les features agrégées (mesures du banc résumées).
#    - DIMENSION produit (dim_produit) : caractéristiques du produit testé
#      (GROUP, NATURE, MODEL) via ItemClassification, clé = PRODUCT_NUMBER.

#  FEATURE ENGINEERING (résumé statistique de chaque séquence de capteur) :
#    mean, std, min, max, med par capteur. C'est ce qui transforme des
#    séquences temporelles hétérogènes (cf. diagnostic Q3) en un vecteur de
#    taille fixe exploitable par un modèle à base d'arbres.

#  features de fuite risque pour entrainement du modèle :
#    calcule aussi durée / nb_mesures mais marqués comme fuite
#    (elles décrivent le déroulement du test, pas l'état du moteur). Le script
#    de modélisation les exclut. On les garde en gold pour transparence/audit.


from pyspark.sql import SparkSession, functions as F

BASE = r"C:\Users\Lisa\Desktop\ESSIN M2\Projet data IA\Projets_Mise_en_situation_2026\Projets_Mise_en_situation_2026\Test Bench"
LAKE = BASE + r"\lakehouse"

spark = (SparkSession.builder.appName("gold")
         .config("spark.sql.shuffle.partitions", "200").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

SENSORS = ["PR_1", "PR_2", "PR_3", "PR_4", "TEMP_1", "TEMP_2", "BRAKE_1",
           "SPEED_1", "SENSOR_1", "SENSOR_2", "SENSOR_4", "SENSOR_5"]

ts = spark.read.parquet(LAKE + r"\silver\timeseries_clean")
res = spark.read.parquet(LAKE + r"\silver\results_dedup")
ic = spark.read.parquet(LAKE + r"\silver\itemclassification_clean")

#  FEATURE ENGINEERING : agrégation par moteur 
agg = []
for c in SENSORS:
    agg += [F.mean(c).alias(f"{c}_mean"), F.stddev(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"), F.max(c).alias(f"{c}_max"),
            F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_med")]

features = (ts.groupBy("SERIAL_NUMBER")
            .agg(*agg,
                 #  métadonnées de test = FUITE potentielle (exclues du ML) 
                 F.count("*").alias("_leak_nb_mesures"),
                 F.countDistinct("PHASE").alias("_leak_nb_programmes"),
                 (F.max("TS").cast("long")
                  - F.min("TS").cast("long")).alias("_leak_duree_sec")))

#  TABLE DE FAITS : features + cible + clés de dimension 
fait = (features.join(res.select("SERIAL_NUMBER", "LABEL",
                                  "PRODUCT_NUMBER", "TEST_STAND"),
                      "SERIAL_NUMBER", "inner"))
fait.write.mode("overwrite").parquet(LAKE + r"\gold\fait_test_moteur")
n = fait.count(); p = fait.filter("LABEL = 1").count()
print(f"gold/fait_test_moteur : {n} moteurs, {p} positifs ({100*p/n:.1f} %)")

#  DIMENSION produit 
dim_produit = ic.select(
    F.col("PRODUCT_NUMBER"),
    *[F.col(c) for c in ["GROUP", "NATURE", "MODEL", "STATUS"] if c in ic.columns]
).dropDuplicates(["PRODUCT_NUMBER"])
dim_produit.write.mode("overwrite").parquet(LAKE + r"\gold\dim_produit")
print(f"gold/dim_produit : {dim_produit.count()} produits")

#  Export du dataset ML aplati (fait + dimension) en parquet ET csv 
# parquet pour la performance, csv en secours (lecture universelle).
dataset_ml = fait.join(dim_produit, "PRODUCT_NUMBER", "left")
pdf = dataset_ml.toPandas()
pdf.to_parquet(LAKE + r"\gold\dataset_ml.parquet", index=False)
pdf.to_csv(LAKE + r"\gold\dataset_ml.csv", index=False)
print(f"gold/dataset_ml : {pdf.shape[0]} lignes x {pdf.shape[1]} colonnes")

print("\nCouche GOLD écrite dans", LAKE + r"\gold")
print("Le script de modélisation lit gold/dataset_ml.parquet.")
print("Rappel : les colonnes préfixées '_leak_' sont exclues à la modélisation.")
spark.stop()
