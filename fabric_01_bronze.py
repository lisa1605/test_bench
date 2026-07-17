
#  FABRIC NOTEBOOK — COUCHE BRONZE

#  RÔLE : lire les CSV bruts depuis Files/ et les matérialiser en tables
#         Delta dans la section Tables/ de Lakehouse_Bronze.

from pyspark.sql import functions as F

#  TIMESERIES 
# input_file_name() capture le nom du fichier source : indispensable pour
# extraire le SERIAL_NUMBER (hash 32 hex) dans les couches suivantes.
ts = (spark.read
      .option("header", True)
      .option("sep", ";")
      .csv("Files/TimeSeries_*.csv")
      .withColumn("_source_file", F.input_file_name())
      .withColumn("_ingested_at", F.current_timestamp()))

(ts.write
   .format("delta")
   .mode("overwrite")
   .saveAsTable("bronze_timeseries"))

print(f"bronze_timeseries : {ts.count()} lignes")

#  RESULTS --
res = (spark.read
       .option("header", True)
       .option("sep", ";")
       .csv("Files/Results_201210.csv")
       .withColumn("_ingested_at", F.current_timestamp()))

(res.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("bronze_results"))

print(f"bronze_results : {res.count()} lignes")

#  WARRANTY 
war = (spark.read
       .option("header", True)
       .option("sep", ";")
       .csv("Files/WarrantyResults.csv")
       .withColumn("_ingested_at", F.current_timestamp()))

(war.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("bronze_warranty"))

print(f"bronze_warranty : {war.count()} lignes")

#  ITEMCLASSIFICATION 
ic = (spark.read
      .option("header", True)
      .option("sep", ";")
      .csv("Files/ItemClassification.csv")
      .withColumn("_ingested_at", F.current_timestamp()))

(ic.write
   .format("delta")
   .mode("overwrite")
   .saveAsTable("bronze_itemclassification"))

print(f"bronze_itemclassification : {ic.count()} lignes")
print("\nCouche BRONZE terminée — tables Delta visibles dans Lakehouse_Bronze > Tables")