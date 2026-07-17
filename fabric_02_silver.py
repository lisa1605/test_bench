from pyspark.sql import functions as F, Window
 
# Préfixe pour lire depuis Lakehouse_Bronze (lakehouse secondaire)
BRONZE = "Lakehouse_Bronze"
 
SENSORS = ["PR_1", "PR_2", "PR_3", "PR_4", "TEMP_1", "TEMP_2", "BRAKE_1",
           "SPEED_1", "SENSOR_1", "SENSOR_2", "SENSOR_4", "SENSOR_5"]
# SENSOR_3 retiré : constant (= 0 sur l'ensemble des données, cf. EDA)
 
#  TIMESERIES : nettoyage complet 
ts = spark.read.table(f"{BRONZE}.bronze_timeseries")
 
ts = (ts
      # Extraction du SERIAL_NUMBER depuis le nom de fichier (hash 32 hex)
      .withColumn("SERIAL_NUMBER",
                  F.regexp_extract(F.col("_source_file"),
                                   r"([0-9a-fA-F]{32})", 1))
      # Parsing du timestamp (vérification de l'ordre chronologique)
      .withColumn("TS", F.to_timestamp(
                  F.concat_ws(" ", "DATE", "HEURE"), "dd/MM/yyyy HH:mm:ss")))
 
# Typage numérique des capteurs
for c in SENSORS:
    ts = ts.withColumn(c, F.col(c).cast("double"))
 
# Tri chronologique par moteur (0.01 % d'inversions détectées en EDA)
ts_clean = (ts
            .select("SERIAL_NUMBER", "TS", "PHASE", *SENSORS)
            .orderBy("SERIAL_NUMBER", "TS"))
 
(ts_clean.write
         .format("delta")
         .mode("overwrite")
         .saveAsTable("silver_timeseries_clean"))
 
n_lignes = ts_clean.count()
n_moteurs = ts_clean.select("SERIAL_NUMBER").distinct().count()
print(f"silver_timeseries_clean : {n_lignes:,} lignes, {n_moteurs} moteurs")
 
#  RESULTS : label + dédoublonnage "pire résultat" -
# 41 moteurs ont été testés plusieurs fois, dont 19 avec résultats
# contradictoires (G et W). Règle retenue : pire résultat fait foi
# (W prime sur G) -> max(LABEL) par moteur.
res = spark.read.table(f"{BRONZE}.bronze_results")
 
res = (res
       .withColumn("SERIAL_NUMBER", F.trim("SERIAL_NUMBER"))
       .withColumn("PRODUCT_NUMBER", F.trim("PRODUCT_NUMBER"))
       .withColumn("LAB", F.when(F.col("RESULT") == "W", 1).otherwise(0)))
 
results_dedup = (res
                 .groupBy("SERIAL_NUMBER")
                 .agg(F.max("LAB").alias("LABEL"),
                      F.count("*").alias("nb_tests"),
                      F.first("PRODUCT_NUMBER").alias("PRODUCT_NUMBER"),
                      F.first("TEST_STAND").alias("TEST_STAND")))
 
(results_dedup.write
              .format("delta")
              .mode("overwrite")
              .saveAsTable("silver_results_dedup"))
 
print(f"\nsilver_results_dedup : {results_dedup.count()} moteurs")
print("Répartition du label (0=G bon, 1=W anomalie) :")
results_dedup.groupBy("LABEL").count().orderBy("LABEL").show()
 
#  ITEMCLASSIFICATION 
ic = spark.read.table(f"{BRONZE}.bronze_itemclassification")
for c in ic.columns:
    if not c.startswith("_"):
        ic = ic.withColumn(c, F.trim(F.col(c)))
 
(ic.write
   .format("delta")
   .mode("overwrite")
   .saveAsTable("silver_itemclassification_clean"))
 
print(f"silver_itemclassification_clean : {ic.count()} produits")
 
#  WARRANTY (documentation uniquement) 
# Audit de jointure : aucun serial de warranty ne correspond à un serial de
# timeseries/results (recouvrement = 0 sur 25 serials, même format 32 hex).
# Conservée dans silver pour transparence et traçabilité de l'audit.
war = spark.read.table(f"{BRONZE}.bronze_warranty")
war = war.withColumn("SERIAL_NUMBER", F.trim("SERIAL_NUMBER"))
 
(war.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("silver_warranty_clean"))
 
print("silver_warranty_clean : conservée pour documentation")
print("  -> recouvrement warranty ∩ results/timeseries = 0 (hors périmètre ML)")
print("\nCouche SILVER terminée — tables Delta visibles dans Lakehouse_Silver > Tables")