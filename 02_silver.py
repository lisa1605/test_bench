
#  MÉDAILLON — COUCHE SILVER : nettoyage & conformité

#  Rôle : appliquer toutes les règles de QUALITÉ et de nettoyage validées
#         pendant l'exploration, pour produire des tables propres et fiables.

#  Transformations (toutes justifiées par l'EDA) :
#   TIMESERIES :
#     - extraction du SERIAL_NUMBER depuis le nom de fichier (hash 32 hex)
#     - typage numérique des capteurs
#     - parsing du timestamp (DATE + HEURE)
#     - suppression de SENSOR_3 (constant = 0, aucune information)
#     - tri chronologique par (serial, timestamp)  [déjà ~trié : 0.01 % d'inversions]
#   RESULTS :
#     - trim des identifiants
#     - LABEL ligne : W=1 (anomalie), G=0 (bon)
#     - DÉDOUBLONNAGE au niveau moteur : règle "PIRE RÉSULTAT" -> un moteur ayant
#       raté au moins une fois est étiqueté 1 (max du label sur ses tests).
#   WARRANTY / ITEMCLASSIFICATION : trim / typage légers.

#  NB : la table warranty est conservée pour documentation, mais l'exploration
#  a montré qu'elle ne partage pas de serial avec results/timeseries
#  (recouvrement = 0). Entre pas dans la cible ML

from pyspark.sql import SparkSession, functions as F

BASE = r"C:\Users\Lisa\Desktop\ESSIN M2\Projet data IA\Projets_Mise_en_situation_2026\Projets_Mise_en_situation_2026\Test Bench"
LAKE = BASE + r"\lakehouse"

spark = (SparkSession.builder.appName("silver")
         .config("spark.sql.shuffle.partitions", "200").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

SENSORS = ["PR_1", "PR_2", "PR_3", "PR_4", "TEMP_1", "TEMP_2", "BRAKE_1",
           "SPEED_1", "SENSOR_1", "SENSOR_2", "SENSOR_4", "SENSOR_5"]  # SENSOR_3 retiré

#  TIMESERIES 
ts = spark.read.parquet(LAKE + r"\bronze\timeseries")
ts = (ts
      .withColumn("SERIAL_NUMBER",
                  F.regexp_extract(F.col("_source_file"), r"([0-9a-fA-F]{32})", 1))
      .withColumn("TS", F.to_timestamp(F.concat_ws(" ", "DATE", "HEURE"),
                                       "dd/MM/yyyy HH:mm:ss")))
for c in SENSORS:
    ts = ts.withColumn(c, F.col(c).cast("double"))
ts_clean = (ts.select("SERIAL_NUMBER", "TS", "PHASE", *SENSORS)
            .orderBy("SERIAL_NUMBER", "TS"))
ts_clean.write.mode("overwrite").parquet(LAKE + r"\silver\timeseries_clean")
print(f"silver/timeseries_clean : {ts_clean.count()} lignes, "
      f"{ts_clean.select('SERIAL_NUMBER').distinct().count()} moteurs")

#  RESULTS : label + dédoublonnage "pire résultat" -
res = spark.read.parquet(LAKE + r"\bronze\results")
res = (res.withColumn("SERIAL_NUMBER", F.trim("SERIAL_NUMBER"))
       .withColumn("PRODUCT_NUMBER", F.trim("PRODUCT_NUMBER"))
       .withColumn("LAB", F.when(F.col("RESULT") == "W", 1).otherwise(0)))

results_dedup = (res.groupBy("SERIAL_NUMBER")
                 .agg(F.max("LAB").alias("LABEL"),               # pire résultat
                      F.count("*").alias("nb_tests"),
                      F.first("PRODUCT_NUMBER").alias("PRODUCT_NUMBER"),
                      F.first("TEST_STAND").alias("TEST_STAND")))
results_dedup.write.mode("overwrite").parquet(LAKE + r"\silver\results_dedup")
print(f"silver/results_dedup : {results_dedup.count()} moteurs")
results_dedup.groupBy("LABEL").count().orderBy("LABEL").show()

#  ITEMCLASSIFICATION (dimension produit) 
ic = spark.read.parquet(LAKE + r"\bronze\itemclassification")
for c in ic.columns:
    if not c.startswith("_"):
        ic = ic.withColumn(c, F.trim(F.col(c)))
ic.write.mode("overwrite").parquet(LAKE + r"\silver\itemclassification_clean")
print("silver/itemclassification_clean écrit")

#  WARRANTY (conservée pour documentation, hors périmètre ML) 
war = spark.read.parquet(LAKE + r"\bronze\warranty")
war = war.withColumn("SERIAL_NUMBER", F.trim("SERIAL_NUMBER"))
war.write.mode("overwrite").parquet(LAKE + r"\silver\warranty_clean")
print("silver/warranty_clean écrit (non utilisée pour la cible : recouvrement nul)")

print("\nCouche SILVER écrite dans", LAKE + r"\silver")
spark.stop()
