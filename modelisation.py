
#  BANC DE TEST MOTEURS — MODÉLISATION & COMPARAISON DE MODÈLES

#  Repart du dataset agrégé produit par l'EDA (eda_figures/dataset_moteur.parquet)
#  Cible : LABEL (1 = W / anomalie, 0 = G / bon) ~25 positifs sur ~1262

#  DÉCISIONS DE CADRAGE (justifiées dans le rapport) :
#   - exclusion des features de fuite (durée du test, nb de mesures, nb programmes)
#     qui décrivent le déroulement du test plutôt que l'état physique du moteur  garde que les features CAPTEURS (comportement physique mesuré)
#   - Déséquilibre 2 % -> class_weight='balanced' + métrique AUC-PR (average precision), JAMAIS l'accuracy
#   - Validation croisée stratifiée (5 folds) : indispensable avec 25 positifs ;
#     un simple train/test laisserait trop peu de positifs en test (trop instable)
#   - Scaling appris dans chaque fold (Pipeline) -> pas de fuite méthodologique
#
#  MODÈLES COMPARÉS :
#   1. Régression logistique (linéaire, sensible à la colinéarité vue en C2)
#   2. Random Forest         (arbres, robuste colinéarité/échelles)
#   3. Gradient Boosting     (arbres boostés, souvent le + performant)
#   4. Isolation Forest      (NON supervisé : les W sont-ils juste des outliers ?) --> nop 


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              IsolationForest)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve)

OUT = "eda_figures"
import os
os.makedirs(OUT, exist_ok=True)

# Charge le dataset GOLD (schéma étoile aplati) parquet de préférence, csv sinon
BASE = r"C:\Users\Lisa\Desktop\ESSIN M2\Projet data IA\Projets_Mise_en_situation_2026\Projets_Mise_en_situation_2026\Test Bench"
GOLD = BASE + r"\lakehouse\gold"
try:
    pdf = pd.read_parquet(GOLD + r"\dataset_ml.parquet")
except Exception:
    pdf = pd.read_csv(GOLD + r"\dataset_ml.csv")

#  Sélection des features : capteurs uniquement, sans fuite 
# Les colonnes de fuite sont désormais préfixées '_leak_' par la couche gold
ID = {"SERIAL_NUMBER", "LABEL", "PRODUCT_NUMBER", "TEST_STAND",
      "GROUP", "NATURE", "MODEL", "STATUS", "nb_tests"}
feat_cols = [c for c in pdf.columns
             if not c.startswith("_leak_") and c not in ID
             and pd.api.types.is_numeric_dtype(pdf[c])]

X = pdf[feat_cols].fillna(pdf[feat_cols].median()).values
y = pdf["LABEL"].astype(int).values
print(f"{X.shape[0]} moteurs, {X.shape[1]} features physiques, "
      f"{y.sum()} positifs ({100*y.mean():.1f} %)")
print("Features utilisées :", feat_cols)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#  Définition des modèles supervisés (pipeline avec scaling interne) 
models = {
    "LogReg": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1))]),
    "GradBoosting": Pipeline([
        ("clf", GradientBoostingClassifier(random_state=42))]),
}

# Baseline de référence : proportion de positifs (AUC-PR d'un modèle aléatoire)
baseline_ap = y.mean()
print(f"\nBaseline AUC-PR (aléatoire) = {baseline_ap:.3f}\n")

results = {}

#  Modèles supervisés en CV stratifiée 
for name, model in models.items():
    ap_scores, roc_scores = [], []
    oof_scores = np.zeros(len(y))            # scores out-of-fold pour courbe PR
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            s = model.predict_proba(X[te])[:, 1]
        else:
            s = model.decision_function(X[te])
        oof_scores[te] = s
        ap_scores.append(average_precision_score(y[te], s))
        roc_scores.append(roc_auc_score(y[te], s))
    results[name] = {
        "AP_mean": np.mean(ap_scores), "AP_std": np.std(ap_scores),
        "ROC_mean": np.mean(roc_scores), "ROC_std": np.std(roc_scores),
        "oof": oof_scores, "supervised": True}
    print(f"{name:14s} AUC-PR = {np.mean(ap_scores):.3f} ± {np.std(ap_scores):.3f}"
          f"   ROC-AUC = {np.mean(roc_scores):.3f} ± {np.std(roc_scores):.3f}")

#  Isolation Forest (NON supervisé) 
# pas dentrainement sur y scoring de l'anomalie et on voit si les W remontent
# Score d'anomalie = -score_samples (plus grand = plus anormal)
iso = Pipeline([("sc", StandardScaler()),
                ("iso", IsolationForest(n_estimators=400,
                                        contamination=float(y.mean()),
                                        random_state=42))])
iso.fit(X)                                    # non supervisé : tout X, sans y
iso_score = -iso.named_steps["iso"].score_samples(iso.named_steps["sc"].transform(X))
ap_iso  = average_precision_score(y, iso_score)
roc_iso = roc_auc_score(y, iso_score)
results["IsolationForest"] = {
    "AP_mean": ap_iso, "AP_std": 0.0, "ROC_mean": roc_iso, "ROC_std": 0.0,
    "oof": iso_score, "supervised": False}
print(f"{'IsolationForest':14s} AUC-PR = {ap_iso:.3f} (non supervisé)"
      f"        ROC-AUC = {roc_iso:.3f}")


#  FIGURES

#  Barres AUC-PR comparées + baseline 
fig, ax = plt.subplots(figsize=(9, 5))
names = list(results.keys())
aps   = [results[n]["AP_mean"] for n in names]
errs  = [results[n]["AP_std"]  for n in names]
colors = ["#4C72B0" if results[n]["supervised"] else "#C44E52" for n in names]
ax.bar(names, aps, yerr=errs, color=colors, capsize=4)
ax.axhline(baseline_ap, ls="--", color="grey",
           label=f"aléatoire ({baseline_ap:.3f})")
ax.set_ylabel("AUC-PR (average precision)")
ax.set_title("Comparaison des modèles — AUC-PR (rouge = non supervisé)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/M1_comparaison_aucpr.png", dpi=120); plt.close(fig)

#  Courbes precision-recall 
fig, ax = plt.subplots(figsize=(8, 6))
for n in names:
    p, r, _ = precision_recall_curve(y, results[n]["oof"])
    ax.plot(r, p, label=f"{n} (AP={results[n]['AP_mean']:.2f})")
ax.axhline(baseline_ap, ls="--", color="grey")
ax.set_xlabel("Recall (part des W retrouvés)")
ax.set_ylabel("Precision")
ax.set_title("Courbes Precision-Recall")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/M2_courbes_pr.png", dpi=120); plt.close(fig)

#  Importance des features (meilleur modèle arbres) 
best_tree = RandomForestClassifier(n_estimators=400,
                                   class_weight="balanced_subsample",
                                   random_state=42, n_jobs=-1).fit(X, y)
imp = pd.Series(best_tree.feature_importances_, index=feat_cols).sort_values()
fig, ax = plt.subplots(figsize=(8, 9))
imp.tail(20).plot(kind="barh", ax=ax)
ax.set_title("Importance des features (Random Forest)")
fig.tight_layout(); fig.savefig(f"{OUT}/M3_importances.png", dpi=120); plt.close(fig)


