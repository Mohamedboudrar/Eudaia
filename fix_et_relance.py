"""
fix_et_relance.py
-----------------
Lance ce script UNE SEULE FOIS depuis le dossier qvt_project.
Il supprime l'ancien modèle et en crée un nouveau compatible avec ta version de scikit-learn.

  python fix_et_relance.py
"""

import os, sys, pickle

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "theme_classifier.pkl")

# ── 1. Supprimer l'ancien .pkl ────────────────────────────────────────────────
if os.path.exists(MODEL_FILE):
    os.remove(MODEL_FILE)
    print("🗑️  Ancien modèle supprimé.")
else:
    print("ℹ️  Pas de modèle existant, on repart de zéro.")

# ── 2. Ré-entraîner avec la version locale de scikit-learn ────────────────────
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

DATA_DIR      = os.path.join(BASE_DIR, "data")
VERBATIMS_CSV = os.path.join(DATA_DIR, "verbatims_entrainement.csv")

def nettoyer(texte):
    texte = texte.lower().strip()
    texte = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüç]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte

print("⏳ Chargement des données d'entraînement...")
df = pd.read_csv(VERBATIMS_CSV, encoding="utf-8")
df = df.dropna(subset=["texte", "theme"])
df["texte_propre"] = df["texte"].apply(nettoyer)

X = df["texte_propre"]
y = df["theme"]

print(f"   {len(df)} phrases chargées · {y.nunique()} thèmes : {sorted(y.unique())}")

print("⏳ Entraînement du nouveau modèle...")

# Pipeline sans multi_class (paramètre supprimé dans scikit-learn >= 1.5)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
    ))
])

if len(X) > 20:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n📊 Performances :")
    print(classification_report(y_test, y_pred))
else:
    pipeline.fit(X, y)

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Nouveau modèle sauvegardé → {MODEL_FILE}")

# ── 3. Test rapide ────────────────────────────────────────────────────────────
print("\n🧪 Test rapide :")
tests = [
    "Je suis débordé, je travaille jusqu'à 21h tous les soirs.",
    "Mon manager ne nous donne jamais de retour sur notre travail.",
    "Le logiciel plante constamment, je perds un temps fou.",
    "Super ambiance dans l'équipe, on s'entraide vraiment bien.",
    "Je n'arrive plus à dormir, je pense à démissionner tellement je suis à bout.",
    "On m'a promis une augmentation et elle n'est jamais arrivée.",
    "Je passe 3h dans les transports alors que je pourrais télé travailler.",
]

for t in tests:
    t_propre = nettoyer(t)
    probas  = pipeline.predict_proba([t_propre])[0]
    classes = pipeline.classes_
    idx_max = probas.argmax()
    theme   = classes[idx_max]
    conf    = probas[idx_max]
    print(f"  [{theme:12}] conf={conf:.2f}  →  {t[:65]}")

print("\n✅ Tout est bon ! Lance maintenant : python app.py")
