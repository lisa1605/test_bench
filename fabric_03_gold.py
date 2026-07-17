#  FABRIC NOTEBOOK — COUCHE GOLD
#  PRÉREQUIS : avoir exécuté Bronze puis Silver.
#  Attacher CE notebook à Lakehouse_Gold (lakehouse par défaut).
#  Ajouter Lakehouse_Silver en tant que lakehouse SECONDAIRE.
#
#  RÔLE : produire le schéma en étoile et le dataset ML final.
#
#  SCHÉMA EN ÉTOILE :
#    - TABLE DE FAITS  gold_fait_test_moteur :
#        une ligne = un moteur
#        colonnes = LABEL (cible) + 60 features capteurs + clés de dimension
#        colonnes _leak_ = features de fuite (calculées mais exclues du ML)
#    - DIMENSION       gold_dim_produit :
#        27 produits issus d'ItemClassification
#        NOTE : couvre seulement 27/177 produits testés (limite des données
#        fournies, documentée)
#
#  FEATURE ENGINEERING :
#    Résumé statistique de chaque séquence capteur par moteur :
#    mean, std, min, max, med -> 12 capteurs x 5 stats = 60 features.
#    Justification : diagnostic Q3 montre des formes hétérogènes entre
#    séquences (var > 0.6) -> agrégation > séquentiel (LSTM écarté).

from pyspark.sql import functions as F

SILVER = "Lakehouse_Silver"

SENSORS = ["PR_1", "PR_2", "PR_3", "PR_4", "TEMP_1", "TEMP_2", "BRAKE_1",
           "SPEED_1", "SENSOR_1", "SENSOR_2", "SENSOR_4", "SENSOR_5"]

ts  = spark.read.table(f"{SILVER}.silver_timeseries_clean")
res = spark.read.table(f"{SILVER}.silver_results_dedup")
ic  = spark.read.table(f"{SILVER}.silver_itemclassification_clean")

#  FEATURE ENGINEERING : agrégation par moteur 
agg = []
for c in SENSORS:
    agg += [
        F.mean(c).alias(f"{c}_mean"),
        F.stddev(c).alias(f"{c}_std"),
        F.min(c).alias(f"{c}_min"),
        F.max(c).alias(f"{c}_max"),
        F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_med")]

features = (ts.groupBy("SERIAL_NUMBER")
              .agg(*agg,
                   # Features de fuite : décrivent le déroulement du test,
                   # pas l'état du moteur. Préfixe _leak_ = exclues du ML.
                   F.count("*").alias("_leak_nb_mesures"),
                   F.countDistinct("PHASE").alias("_leak_nb_programmes"),
                   (F.max("TS").cast("long")
                    - F.min("TS").cast("long")).alias("_leak_duree_sec")))

#  TABLE DE FAITS -
fait = (features.join(
            res.select("SERIAL_NUMBER", "LABEL",
                       "PRODUCT_NUMBER", "TEST_STAND"),
            "SERIAL_NUMBER", "inner"))

(fait.write
     .format("delta")
     .mode("overwrite")
     .saveAsTable("gold_fait_test_moteur"))

n = fait.count(); p = fait.filter("LABEL = 1").count()
print(f"gold_fait_test_moteur : {n} moteurs, {p} positifs ({100*p/n:.1f} %)")

#  DIMENSION PRODUIT 
# Sélection des colonnes disponibles (défensif selon les données réelles)
dim_cols = [c for c in ["PRODUCT_NUMBER", "GROUP", "NATURE", "MODEL", "STATUS"]
            if c in ic.columns]
dim_produit = ic.select(*dim_cols).dropDuplicates(["PRODUCT_NUMBER"])

(dim_produit.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable("gold_dim_produit"))

print(f"gold_dim_produit : {dim_produit.count()} produits")
print("  NOTE : ItemClassification couvre 27/177 produits testés.")
print("  La dimension est conservée à titre indicatif (non utilisée en ML).")

#  VÉRIFICATION DU SCHÉMA ÉTOILE 
print("\n=== VÉRIFICATION FINALE ===")
print(f"Grain table de faits unique : "
      f"{fait.select('SERIAL_NUMBER').distinct().count() == fait.count()}")
print(f"LABEL nuls : {fait.filter(F.col('LABEL').isNull()).count()}")
print(f"Colonnes _leak_ isolées : "
      f"{[c for c in fait.columns if c.startswith('_leak_')]}")
print(f"Nb features ML (hors _leak_, hors clés) : "
      f"{len([c for c in fait.columns if not c.startswith('_leak_') and c not in ('SERIAL_NUMBER','LABEL','PRODUCT_NUMBER','TEST_STAND')])}")

print("\nCouche GOLD terminée.")
print("Tables disponibles dans Lakehouse_Gold > Tables :")
print("  - gold_fait_test_moteur  (table de faits ML)")
print("  - gold_dim_produit       (dimension produit, indicative)")
print("\nProchaine étape : notebook de modélisation qui lit gold_fait_test_moteur.")