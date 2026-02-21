"""
nlp_engine.py
-------------
Moteur NLP du projet QVT.
Responsabilités :
  1. Entraîner le classifieur de thèmes (TF-IDF + Logistic Regression)
  2. Calculer le score de sentiment (lexique + règles)
  3. Détecter les signaux faibles burnout
  4. Exposer une fonction analyse(texte) → dict résultat
"""

import os, re, csv, json, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

VERBATIMS_CSV  = os.path.join(DATA_DIR, "verbatims_entrainement.csv")
LEXIQUE_CSV    = os.path.join(DATA_DIR, "lexique_sentiment.csv")
RECOS_JSON     = os.path.join(DATA_DIR, "recommandations.json")
MODEL_FILE     = os.path.join(MODEL_DIR, "theme_classifier.pkl")

# ── Mots déclencheurs burnout ─────────────────────────────────────────────────
BURNOUT_TRIGGERS = [
    "burnout", "craquer", "plus de force", "à bout", "plus envie",
    "ras-le-bol", "démissionner", "je veux partir", "je n'en peux plus",
    "vidé", "épuisement total", "plus capable", "arrêt maladie",
    "ne dors plus", "n'arrive plus à dormir"
]

# ── 1. NETTOYAGE ──────────────────────────────────────────────────────────────
def nettoyer(texte: str) -> str:
    """Minuscules + suppression ponctuation lourde, garde les apostrophes."""
    texte = texte.lower().strip()
    texte = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüç]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte

# ── 2. ENTRAÎNEMENT DU CLASSIFIEUR DE THÈMES ─────────────────────────────────
def entrainer_modele(force=False):
    """
    Charge verbatims_entrainement.csv, entraîne TF-IDF + LogReg,
    sauvegarde le modèle dans models/theme_classifier.pkl.
    Si le modèle existe déjà, ne ré-entraîne pas (sauf force=True).
    """
    if os.path.exists(MODEL_FILE) and not force:
        return charger_modele()

    print("⏳ Entraînement du modèle de classification thèmes...")
    df = pd.read_csv(VERBATIMS_CSV, encoding="utf-8")
    df = df.dropna(subset=["texte", "theme"])
    df["texte_propre"] = df["texte"].apply(nettoyer)

    X = df["texte_propre"]
    y = df["theme"]

    # Pipeline : TF-IDF (1-grammes et 2-grammes) + Logistic Regression
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
            class_weight="balanced",   # compense les classes déséquilibrées
            solver="lbfgs",
        ))
    ])

    # Split train/test pour afficher les perfs
    if len(X) > 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print("\n📊 Performances du classifieur :")
        print(classification_report(y_test, y_pred))
    else:
        pipeline.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"✅ Modèle sauvegardé → {MODEL_FILE}")
    return pipeline

def charger_modele():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

# ── 3. LEXIQUE SENTIMENT ──────────────────────────────────────────────────────
def charger_lexique() -> dict:
    """Retourne {mot: (score, type)} depuis lexique_sentiment.csv"""
    lexique = {}
    with open(LEXIQUE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lexique[row["mot"].lower().strip()] = {
                "score": float(row["score"]),
                "type": row["type"]
            }
    return lexique

LEXIQUE = charger_lexique()

def calculer_sentiment(texte: str) -> tuple[float, str]:
    """
    Calcule le score de sentiment d'un texte.
    Applique les intensificateurs sur le mot suivant.
    Gère la négation simple (ne...pas, pas, jamais, aucun).
    Retourne (score_float, label_string)
    """
    tokens = nettoyer(texte).split()
    score_total = 0.0
    nb_mots_sentiments = 0
    intensificateur_actif = 1.0
    negation_active = False

    # Mots de négation
    NEGATIONS = {"pas", "ne", "plus", "jamais", "aucun", "rien", "sans", "ni"}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Vérif négation
        if token in NEGATIONS:
            negation_active = True
            i += 1
            continue

        # Vérif bigram (ex: "un peu", "plus de")
        bigram = token + " " + tokens[i+1] if i + 1 < len(tokens) else ""
        mot_a_tester = bigram if bigram in LEXIQUE else token

        if mot_a_tester in LEXIQUE:
            entree = LEXIQUE[mot_a_tester]

            if entree["type"] == "intensificateur":
                intensificateur_actif = entree["score"]
                if mot_a_tester == bigram:
                    i += 1  # skip le 2ème token du bigram
            else:
                s = entree["score"] * intensificateur_actif
                if negation_active:
                    s = -s * 0.8   # inversion partielle
                score_total += s
                nb_mots_sentiments += 1
                # Reset après usage
                intensificateur_actif = 1.0
                negation_active = False

        i += 1

    if nb_mots_sentiments == 0:
        return 0.0, "NEUTRE"

    score_moyen = max(-1.0, min(1.0, score_total / nb_mots_sentiments))

    if score_moyen <= -0.3:
        label = "NEGATIF"
    elif score_moyen >= 0.3:
        label = "POSITIF"
    else:
        label = "NEUTRE"

    return round(score_moyen, 4), label

# ── 4. DÉTECTION SIGNAL FAIBLE BURNOUT ───────────────────────────────────────
def detecter_burnout(texte: str, score_sentiment: float) -> bool:
    """
    Retourne True si le texte contient des signaux de burnout.
    Critères : mot déclencheur présent ET score sentiment < -0.55
    """
    texte_low = texte.lower()
    trigger_trouve = any(trigger in texte_low for trigger in BURNOUT_TRIGGERS)
    return trigger_trouve and score_sentiment < -0.55

# ── 5. FONCTION PRINCIPALE : analyse complète d'un verbatim ──────────────────
_modele = None  # cache du modèle en mémoire

def analyser(texte: str) -> dict:
    """
    Point d'entrée principal.
    Reçoit un texte brut, retourne un dict avec thème + sentiment + score + burnout.
    Le texte brut N'EST PAS retourné dans le résultat (anonymat).
    """
    global _modele
    if _modele is None:
        if os.path.exists(MODEL_FILE):
            _modele = charger_modele()
        else:
            _modele = entrainer_modele()

    texte_propre = nettoyer(texte)

    # ── Détection du thème
    probas = _modele.predict_proba([texte_propre])[0]
    classes = _modele.classes_
    idx_max = probas.argmax()
    theme_predit = classes[idx_max]
    confiance = round(float(probas[idx_max]), 4)

    # Si confiance trop faible, on ne classe pas
    if confiance < 0.15:
        theme_predit = "NON_CLASSE"

    # ── Sentiment
    score_sent, label_sent = calculer_sentiment(texte)

    # ── Signal burnout
    signal_burnout = detecter_burnout(texte, score_sent)

    # ── Résultat (SANS le texte brut)
    return {
        "theme":          theme_predit,
        "confiance":      confiance,
        "sentiment":      label_sent,
        "score_sentiment": score_sent,
        "signal_burnout": signal_burnout,
    }

# ── 6. RECOMMANDATIONS ───────────────────────────────────────────────────────
def charger_recommandations() -> dict:
    with open(RECOS_JSON, encoding="utf-8") as f:
        return json.load(f)

def get_recommandation(theme: str, pct_negatif: float) -> dict | None:
    """
    Retourne la recommandation pour un thème si pct_negatif > seuil (60%).
    """
    recos = charger_recommandations()
    if theme not in recos:
        return None
    reco = recos[theme]
    if pct_negatif >= reco["seuil"]:
        return reco
    return None

# ── Script standalone : entraîne le modèle si lancé directement ──────────────
if __name__ == "__main__":
    entrainer_modele(force=True)
    print("\n🧪 Test rapide :")
    tests = [
        "Je suis complètement débordé, je travaille jusqu'à 21h tous les soirs.",
        "Mon manager ne nous donne jamais de retour, c'est très frustrant.",
        "Les outils plantent constamment, je perds un temps fou.",
        "Super ambiance dans l'équipe, on s'entraide vraiment bien.",
        "Je n'arrive plus à dormir tellement je pense au travail.",
        "On m'a promis une augmentation et elle n'est jamais arrivée.",
    ]
    for t in tests:
        r = analyser(t)
        burnout = "🚨 BURNOUT" if r["signal_burnout"] else ""
        print(f"  [{r['theme']:12}] [{r['sentiment']:8}] score={r['score_sentiment']:+.2f}  {burnout}")
        print(f"   → {t[:70]}")
        print()
