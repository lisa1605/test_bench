# PROJET TESTBENCH  Banc de Test Moteurs Pipeline de Données & Détection d'Anomalies

## Contexte

Ce projet analyse les données issues de bancs de tests industriels de moteurs électriques. L'objectif de ce projet est la mise en place d’un système de collecte de données dans le domaine de l’industrie et détecter les moteurs produisant un résultat anormal à partir des mesures physiques enregistrées pendant le test, en construisant un pipeline de données complet et un modèle de machine learning supervisé.

Les conditions imposent que la récolte des données soit faite en batch initialement puis en streaming par la suite. 

## Choix d'architecture : pourquoi le Lakehouse médaillon ?

Nous avons choisi une architecture Lakehouse car elle permet de traiter les données brutes et d'obtenir des tables fiables suite à nos analyses. On anticipe aussi le traitement des données streaming en choisissant le Lakehouse. Il est aussi adapté au Machine Learning et au BI, deux techniques que nous utiliserons par la suite. 
Le grand nombre de fichiers CSV de séries temporelles  hétérogènes justifie aussi ce choix, le lakehouse ingère les fichiers bruts tels quels en Bronze et la transformation peut se faire progressivement à travers les étapes de la méthode en Médaillon. 
Spark sur Frabric traite les données en parallèle, couche par couche sans tout charger en RAM et garantit le traitement des 414064 lignes de mesures. Nous avons effectué les transformation directement dans le lakehouse (silver). J'ai ainsi effectué le principe d'ELT (extract-load-transform) plutôt qu'ETL. 

Nous avons aussi choisi l'architecture médaillon parce qu'elle est adaptée pour le passage en temps réel, d'ailleurs pour ce traitement streaming nous avons aussi prévu d'insérer un bus d'événement Kafka avant l'entrée des données dans le Lakehouse. 
La donnée passera donc d'abord par Kafka puis Bronze puis Silver et enfin Gold en continu. La couche Bronze peut recevoir des données en streaming sans modifier les couches suivantes.  On a pensé toute cette architecture pour évoluer. 

### Pourquoi trois lakehouses séparés ?

L'architecture médaillon impose une séparation stricte des couches  : Bronze, Silver et Gold. 

Les ingénieurs data accèdent à Bronze, les data analysts à Silver et Gold. On ne risque pas d'écraser des données brutes depuis une couche aval et si la logique Gold change (nouvelles features, nouvelle règle métier), on relance uniquement le notebook Gold sans re-ingérer les 414 064 lignes depuis les CSV.
Chaque couche conserve un état stable et daté. Les tables Delta permettent le time travel, on peut revenir à l'état d'hier si une transformation introduit une erreur.

## Architecture technique

Workspace Fabric : test_bench
│
├── bronze_lakehouse        ← Couche BRONZE (ingestion)
│   ├── Files/              ← CSV sources uploadés tels quels
│   └── Tables/            
│       ├── bronze_timeseries         
│       ├── bronze_results           
│       ├── bronze_warranty          
│       └── bronze_itemclassification 
│
├── silver_lakehouse        ← Couche SILVER (nettoyage)
│   └── Tables/
│       ├── silver_timeseries_clean  
│       ├── silver_results_dedup      (1 280 moteurs, label G/W)
│       ├── silver_itemclassification_clean
│       └── silver_warranty_clean    
│
└── gold_lakehouse          ← Couche GOLD (fondation du schéma étoile + ML)
    └── Tables/
        ├── gold_fait_test_moteur    
        └── gold_dim_produit          
```

Nous avons utilisé PySpark sur Microsoft Fabric (Spark managé, sans configuration locale) comme moteur de recherche.
Nous avons choisi le format Delta Lake car c'est une table managée avec un schéma particulier, un historique et des métadonnées (tables transactionnelles, versionnées, requêtables en SQL).  

---

## Étape 1 Bronze : ingestion brute

**Notebook :** `fabric_01_bronze.py`  
**Lakehouse :** `bronze_lakehouse` 

### Rôle
La couche Bronze présente les données brutes, sans aucune transformation appliquée. Elle permet de pouvoir revenir vérifier sur nos données brutes si quelque chose ne va pas dans notre pipeline. Si une hypothèse de transformation est fausse on peut repartir des données originales. 

### Ce qui est fait

Les quatre sources CSV sont lues depuis `Files/` et écrites comme tables Delta dans `Tables/`. Deux colonnes de traçabilité sont ajoutées à chaque table :

- `_source_file` : chemin complet du fichier source colonne critique pour les timeseries, car c'est le seul endroit où figure le `SERIAL_NUMBER` (hash 32 hex dans le nom de fichier, ex. `TimeSeries_20121001_071145_b4f0944f...csv`).
- `_ingested_at` : horodatage d'ingestion, pour l'audit et le versionnement.


### Résultats

| Table                        | Lignes  |
|------------------------------|---------|
| bronze_timeseries            | 414 064 |
| bronze_results               | 1 327   |
| bronze_warranty              | 48      |
| bronze_itemclassification    | 27      |

---

## Étape 2 Silver : nettoyage et conformité

**Notebook :** `fabric_02_silver.py`  
**Lakehouses :** `silver_lakehouse`  + `bronze_lakehouse` 

### Rôle

La couche Silver applique toutes les règles de qualité et de conformité validées pendant l'analyse exploratoire. Elle produit des tables propres, typées, dédoublonnées, prêtes pour l'agrégation.

### Transformations appliquées

**Timeseries (`silver_timeseries_clean`) :**

- Extraction du `SERIAL_NUMBER` depuis `_source_file` par regex `([0-9a-fA-F]{32})` — c'est la clé qui relie chaque séquence à son moteur.
- Parsing du timestamp (`DATE` + `HEURE` → colonne `TS` au format `dd/MM/yyyy HH:mm:ss`).
- Typage numérique explicite des 12 capteurs retenus (cast `double`).
- Suppression de `SENSOR_3` : colonne constante (= 0 sur l'ensemble des données), sans information pour un modèle ML. Constaté lors de l'EDA (Phase A).
- Tri chronologique par `(SERIAL_NUMBER, TS)` : le diagnostic Q1 a montré 0,01 % d'inversions temporelles (mesures à la même seconde), traité par ce tri.

**Resultats (`silver_results_dedup`) :**

Deux points découverts lors de l'audit des données :
- 41 moteurs ont été testés plusieurs fois → plusieurs lignes pour un même `SERIAL_NUMBER`. 
- 19 d'entre eux ont des résultats contradictoires (G sur un test, W sur un autre).

Il est possible que le même moteur ai été testé 2 fois pour vérifier qu'il soit vraiment défectueux? 

Règle métier retenue : le pire résultat fait foi. Un moteur ayant raté au moins un test est étiqueté W (LABEL = 1). Justification : un moteur qui a nécessité un re-test a montré une anomalie détectable au banc c'est  ce qu'on veut apprendre à identifier et prédire par la suite avec le machine learning. 

Implémentation : `F.max("LAB").groupBy("SERIAL_NUMBER")` 
 le max du label binaire (W=1, G=0) retient le W si présent.

**Warranty (`silver_warranty_clean`) :**

Conservée pour traçabilité, mais hors périmètre ML. L'audit de jointure a établi un recouvrement nul entre les 25 serials de warranty et les 1 280 serials de results/timeseries (même format 32 hex, zéro correspondance). Le label « défaillant en usage » initialement prévu n'est donc pas constructible avec ces données, constat documenté, pas un bug.

### Résultats

| Table                           | Moteurs | Positifs (W) |
|---------------------------------|---------|--------------|
| silver_timeseries_clean         | 1 269   | —            |
| silver_results_dedup            | 1 280   | 25 (2,0 %)   |

---

## Étape 3 — Gold : fondation du schéma étoile et feature engineering

**Notebook :** `fabric_03_gold.py`  
**Lakehouses :** `gold_lakehouse`  + `silver_lakehouse` 

### Rôle

La couche Gold produit le modèle sémantique ( fondation du schéma en étoile) et le dataset ML final. C'est la couche qui sera consommée par les analyses, les dashboards et les modèles.

### Schéma en étoile

**Table de faits `gold_fait_test_moteur`**

une ligne = un moteur
 Contient la colonne cible (`LABEL`) et 60 features issues du feature engineering, plus les clés de dimension (`PRODUCT_NUMBER`, `TEST_STAND`).

**Dimension produit `gold_dim_produit`**

Issue d'ItemClassification : `GROUP`, `NATURE`, `MODEL`, `STATUS`. 
Cette dimension ne couvre que 27 des 177 produits testés : audit de jointure effectué, limite des données fournies documentée et assumée. La dimension est conservée à titre indicatif mais n'est pas mobilisée dans la modélisation.

### Feature engineering

Le diagnostic séquentiel (Q3) a montré que les séquences temporelles ne partagent pas de pattern commun (variabilité inter-séquences > 0,6 sur 5 signaux sur 6). Le LSTM et l'autoencoder séquentiel sont donc écartés, sans forme temporelle commune, un modèle de reconstruction ne discrimine pas les anomalies des moteurs normaux.

Les séquences sont résumées en un vecteur de 60 features par moteur :

- **5 statistiques** × **12 capteurs** = 60 features
- Statistiques : `mean`, `std`, `min`, `max`, `med` (percentile 50 %)
- Capteurs retenus : `PR_1..4`, `TEMP_1..2`, `BRAKE_1`, `SPEED_1`, `SENSOR_1..2`, `SENSOR_4..5`

Les features de fuite (`_leak_nb_mesures`, `_leak_nb_programmes`, `_leak_duree_sec`) sont calculées mais **préfixées `_leak_`**, elles sont isolées et non utilisées dans la modélisation. Elles décrivent le déroulement du test (durée, nombre de mesures) plutôt que l'état physique du moteur, et créeraient une fuite de données si incluses.

### Vérification de la table de faits

| Contrôle                         | Résultat |
|----------------------------------|----------|
| Grain unique (1 ligne / moteur)  | ✅ True   |
| LABEL nuls                       | ✅ 0      |
| Valeurs manquantes sur features  | ✅ aucune |
| Colonnes `_leak_` isolées        | ✅ 3      |
| Nb features ML                   | ✅ 60     |

### Résultats

| Table                  | Lignes | Positifs |
|------------------------|--------|----------|
| gold_fait_test_moteur  | 1 262  | 25 (2 %) |
| gold_dim_produit       | 27     | —        |

---

## Résultats de modélisation

Comparaison de 4 modèles en validation croisée stratifiée (5 folds), métrique AUC-PR (adaptée au déséquilibre de classes) :

| Modèle           | AUC-PR          | ROC-AUC         |
|------------------|-----------------|-----------------|
| Baseline aléatoire | 0,020         | —               |
| Régression logistique | 0,338 ± 0,178 | 0,808 ± 0,113 |
| **Random Forest**    | **0,596 ± 0,168** | **0,963 ± 0,018** |
| Gradient Boosting | 0,594 ± 0,233  | 0,898 ± 0,117   |
| Isolation Forest  | 0,086          | 0,827           |

**Modèle retenu : Random Forest** (`class_weight='balanced_subsample'`, 400 estimateurs).

Nous avons choisi ce modèle car sa performance est équivalente au Gradient Boosting, sa variance est nettement plus faible entre folds (±0,168 vs ±0,233), et critique avec seulement 25 positifs. Les features les plus discriminantes sont `SPEED_1_std`, `SENSOR_4_std`, `TEMP_1_std`, les moteurs anormaux se distinguent par une instabilité plus grande de leurs signaux, pas par des niveaux moyens différents.

---

## Limites et perspectives

**Limites identifiées :**
- 25 cas positifs sur 1 262 : les intervalles de confiance sont larges, la robustesse demanderait davantage de données de défaut.
- Recouvrement nul entre warranty et les données de test : le lien entre défaillances terrain et mesures de banc n'est pas constructible avec les données fournies.
- Dimension produit partielle (27/177 produits).

**Évolution vers le streaming (non implémentée) :**
L'architecture médaillon est nativement compatible avec un passage en temps réel. En production, un flux Kafka alimenterait la couche Bronze en continu, Silver et Gold seraient mis à jour par micro-batch, et le modèle serait réentraîné automatiquement selon une planification (ex. quotidienne à 8h via un pipeline Fabric).