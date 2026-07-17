
#  MÉDAILLON  BRONZE : ingestion des données brutes

#  Rôle : lire les CSV SOURCES tels quels et les matérialiser en parquet,
#         SANS transformation métier.ajout de  la TRAÇABILITÉ
#         (fichier source) car elle porte le serial des timeseries.


from pyspark.sql import SparkSession, functions as F

BASE = r"C:\Users\Lisa\Desktop\ESSIN M2\Projet data IA\Projets_Mise_en_situation_2026\Projets_Mise_en_situation_2026\Test Bench"
LAKE = BASE + r"\lakehouse"

spark = (SparkSession.builder.appName("bronze")
         .config("spark.sql.shuffle.partitions", "200").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

#TIMESERIES : on conserve le nom de fichier source (porte le serial) 
ts = (spark.read.option("header", True).option("sep", ";")
      .csv(BASE + r"\TimeSeries\*.csv")
      .withColumn("_source_file", F.input_file_name())
      .withColumn("_ingested_at", F.current_timestamp()))
ts.write.mode("overwrite").parquet(LAKE + r"\bronze\timeseries")
print(f"bronze/timeseries : {ts.count()} lignes")

#RESULTS (a un en-tête) 
res = (spark.read.option("header", True).option("sep", ";")
       .csv(BASE + r"\Results\*.csv")
       .withColumn("_ingested_at", F.current_timestamp()))
res.write.mode("overwrite").parquet(LAKE + r"\bronze\results")
print(f"bronze/results : {res.count()} lignes")

#WARRANTY 
war = (spark.read.option("header", True).option("sep", ";")
       .csv(BASE + r"\Warranty\*.csv")
       .withColumn("_ingested_at", F.current_timestamp()))
war.write.mode("overwrite").parquet(LAKE + r"\bronze\warranty")
print(f"bronze/warranty : {war.count()} lignes")

#ITEMCLASSIFICATION (dimension produit)
ic = (spark.read.option("header", True).option("sep", ";")
      .csv(BASE + r"\ItemClassification\*.csv")
      .withColumn("_ingested_at", F.current_timestamp()))
ic.write.mode("overwrite").parquet(LAKE + r"\bronze\itemclassification")
print(f"bronze/itemclassification : {ic.count()} lignes")

print("\nCouche BRONZE écrite dans", LAKE + r"\bronze")
spark.stop()
