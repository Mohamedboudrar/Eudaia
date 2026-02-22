"""
nlp_engine.py — QVT Agent v3
------------------------------
CORRECTIFS CRITIQUES :
  1. Seuils sentiment resserrés : NEUTRE = [-0.15, +0.15] (était ±0.3 → trop large)
     Avant : score +0.25 = NEUTRE  |  Après : score +0.25 = POSITIF
  2. Lexique enrichi avec TOUS les termes QVT courants (ambiance, équipe, etc.)
  3. Fallback note_quanti : si le score lexical est 0.0 (mot inconnu),
     la note chiffrée (1-5) est utilisée pour inférer le sentiment
  4. Log de debug pour tracer chaque analyse (affiché dans la console Flask)
  5. Normalisation defensive du sentiment avant stockage
"""

import os, re, csv, json, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Optional

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
CACHE_DIR     = os.path.join(MODEL_DIR, "zeroshot_cache")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

VERBATIMS_CSV = os.path.join(DATA_DIR, "verbatims_entrainement.csv")
LEXIQUE_CSV   = os.path.join(DATA_DIR, "lexique_sentiment.csv")
RECOS_JSON    = os.path.join(DATA_DIR, "recommandations.json")
MODEL_LR_FILE = os.path.join(MODEL_DIR, "theme_classifier.pkl")
MODEL_NB_FILE = os.path.join(MODEL_DIR, "theme_classifier_nb.pkl")

MODELS_DISPONIBLES  = ["logistic", "naive_bayes", "zeroshot"]
_ACTIVE_MODEL_FILE  = os.path.join(MODEL_DIR, "active_model.txt")

# ── SEUILS SENTIMENT ──────────────────────────────────────────────────────────
# CORRECTIF CRITIQUE : seuils resserrés pour mieux différencier NEUTRE/POSITIF/NEGATIF
# Ancien : NEUTRE = score dans [-0.30, +0.30]  → trop large, avalait les positifs faibles
# Nouveau : NEUTRE = score dans [-0.15, +0.15] → discrimination plus fine
SEUIL_NEGATIF = -0.15   # en dessous → NEGATIF
SEUIL_POSITIF =  0.15   # au dessus  → POSITIF
# Entre les deux → NEUTRE

# ── Fallback quand le lexique n'a aucun mot ───────────────────────────────────
# Si le score lexical est 0.0 ET une note_quanti est disponible, on l'utilise
# note 1-2 → NEGATIF | note 3 → NEUTRE | note 4-5 → POSITIF
def _sentiment_depuis_note(note: int) -> tuple[float, str]:
    if note <= 2:
        return -0.5, "NEGATIF"
    elif note == 3:
        return 0.0, "NEUTRE"
    else:
        return 0.5, "POSITIF"

# ── Moteur persisté ───────────────────────────────────────────────────────────
def _lire_modele_persistant() -> str:
    try:
        with open(_ACTIVE_MODEL_FILE, "r") as f:
            val = f.read().strip()
            if val in MODELS_DISPONIBLES:
                return val
    except FileNotFoundError:
        pass
    return "logistic"

def _ecrire_modele_persistant(model_type: str):
    os.makedirs(os.path.dirname(_ACTIVE_MODEL_FILE), exist_ok=True)
    with open(_ACTIVE_MODEL_FILE, "w") as f:
        f.write(model_type)

_model_lr       = None
_model_nb       = None
_model_zeroshot = None

# ── Configuration zero-shot ───────────────────────────────────────────────────
ZEROSHOT_MODEL_NAME     = "joeddav/xlm-roberta-large-xnli"
ZEROSHOT_MODEL_FALLBACK = "cross-encoder/nli-MiniLM2-L6-H768"

THEMES_ZEROSHOT: dict[str, str] = {
    "CHARGE_TRAVAIL": "surcharge de travail, heures supplémentaires ou épuisement professionnel",
    "MANAGEMENT":     "management, relations avec le responsable ou le manager",
    "COMMUNICATION":  "communication interne, échanges d'informations ou transparence",
    "AMBIANCE":       "ambiance d'équipe, relations entre collègues ou cohésion",
    "OUTILS":         "outils informatiques, logiciels ou équipements de travail",
    "REMUNERATION":   "salaire, rémunération, augmentation ou avantages financiers",
    "EVOLUTION":      "évolution de carrière, formation professionnelle ou perspectives",
    "EQUILIBRE_VIE":  "équilibre entre vie professionnelle et vie personnelle",
    "RECONNAISSANCE": "reconnaissance au travail, valorisation ou feedback positif",
    "CONDITIONS":     "conditions de travail, locaux, bruit ou ergonomie",
}

SEUIL_CONFIANCE_ZEROSHOT = 0.20
SEUIL_CONFIANCE_TFIDF    = 0.15

BURNOUT_TRIGGERS = [
    "burnout", "craquer", "plus de force", "à bout", "plus envie",
    "ras-le-bol", "démissionner", "je veux partir", "je n'en peux plus",
    "vidé", "épuisement total", "plus capable", "arrêt maladie",
    "ne dors plus", "n'arrive plus à dormir"
]

# =============================================================================
# NETTOYAGE
# =============================================================================

def nettoyer(texte: str) -> str:
    texte = texte.lower().strip()
    texte = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüç]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte

# =============================================================================
# MODÈLES TF-IDF
# =============================================================================

def entrainer_modele(force=False, model_type="all"):
    if model_type == "zeroshot":
        print("ℹ️  Zero-shot : aucun entraînement nécessaire.")
        return
    if not os.path.exists(VERBATIMS_CSV):
        print(f"⚠️  Fichier d'entraînement introuvable : {VERBATIMS_CSV}")
        return

    df = pd.read_csv(VERBATIMS_CSV, encoding="utf-8")
    df = df.dropna(subset=["texte", "theme"])
    df.loc[:, "texte_propre"] = df["texte"].apply(nettoyer)
    X, y = df["texte_propre"], df["theme"]

    if len(X) > 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    tfidf_params = dict(ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True)

    if model_type in ["logistic", "all"]:
        if not os.path.exists(MODEL_LR_FILE) or force:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs"))
            ])
            pipe.fit(X_train, y_train)
            with open(MODEL_LR_FILE, "wb") as f:
                pickle.dump(pipe, f)
            print(f"✅ Logistic Regression entraîné → {MODEL_LR_FILE}")

    if model_type in ["naive_bayes", "all"]:
        if not os.path.exists(MODEL_NB_FILE) or force:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", MultinomialNB(alpha=1.0))
            ])
            pipe.fit(X_train, y_train)
            with open(MODEL_NB_FILE, "wb") as f:
                pickle.dump(pipe, f)
            print(f"✅ Naive Bayes entraîné → {MODEL_NB_FILE}")


def _charger_modele_tfidf(model_type: str):
    path = MODEL_NB_FILE if model_type == "naive_bayes" else MODEL_LR_FILE
    if not os.path.exists(path):
        entrainer_modele(force=True, model_type=model_type)
    with open(path, "rb") as f:
        return pickle.load(f)


def _classifier_theme_tfidf(texte: str, model_type: str) -> tuple[str, float]:
    global _model_lr, _model_nb
    if model_type == "naive_bayes":
        if _model_nb is None:
            _model_nb = _charger_modele_tfidf("naive_bayes")
        model = _model_nb
    else:
        if _model_lr is None:
            _model_lr = _charger_modele_tfidf("logistic")
        model = _model_lr

    texte_propre = nettoyer(texte)
    probas  = model.predict_proba([texte_propre])[0]
    idx_max = probas.argmax()
    theme   = model.classes_[idx_max]
    score   = round(float(probas[idx_max]), 4)

    if score < SEUIL_CONFIANCE_TFIDF:
        theme = "NON_CLASSE"
    return theme, score

# =============================================================================
# ZERO-SHOT
# =============================================================================

def _charger_zeroshot():
    global _model_zeroshot
    if _model_zeroshot is not None:
        return _model_zeroshot
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("pip install transformers torch sentencepiece")

    for model_name in [ZEROSHOT_MODEL_NAME, ZEROSHOT_MODEL_FALLBACK]:
        try:
            print(f"⏳ Chargement zero-shot : {model_name}")
            _model_zeroshot = pipeline(
                task="zero-shot-classification",
                model=model_name,
                cache_dir=CACHE_DIR,
            )
            print(f"✅ Zero-shot chargé : {model_name}")
            return _model_zeroshot
        except Exception as e:
            print(f"⚠️  Échec {model_name} : {e}")
    raise RuntimeError("Impossible de charger un modèle zero-shot.")


def _classifier_theme_zeroshot(texte: str) -> tuple[str, float]:
    clf = _charger_zeroshot()
    labels_list = list(THEMES_ZEROSHOT.values())
    cles_list   = list(THEMES_ZEROSHOT.keys())
    result = clf(
        sequences=texte,
        candidate_labels=labels_list,
        hypothesis_template="Ce texte parle de {}.",
        multi_label=False,
    )
    best_label = result["labels"][0]
    best_score = result["scores"][0]
    idx        = labels_list.index(best_label)
    theme_key  = cles_list[idx]
    if best_score < SEUIL_CONFIANCE_ZEROSHOT:
        theme_key = "NON_CLASSE"
    return theme_key, round(float(best_score), 4)


def ajouter_theme_zeroshot(cle: str, label_fr: str):
    THEMES_ZEROSHOT[cle] = label_fr
    print(f"✅ Thème ajouté : {cle} → \"{label_fr}\"")


def lister_themes_zeroshot():
    print("\n📋 Thèmes zero-shot :")
    for cle, label in THEMES_ZEROSHOT.items():
        print(f"  {cle:20} → {label}")

# =============================================================================
# SENTIMENT — CORRECTIF CENTRAL
# =============================================================================

def charger_lexique() -> dict:
    lexique = {}
    if not os.path.exists(LEXIQUE_CSV):
        print(f"⚠️  Lexique introuvable : {LEXIQUE_CSV}")
        return lexique
    with open(LEXIQUE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lexique[row["mot"].lower().strip()] = {
                "score": float(row["score"]),
                "type":  row["type"]
            }
    return lexique

LEXIQUE = charger_lexique()


def calculer_sentiment(texte: str, note_quanti: int = None) -> tuple[float, str]:
    """
    Calcule le score de sentiment via lexique + règles.

    CORRECTIFS v3 :
    - Seuils resserrés : POSITIF si score >= +0.15 (était +0.30)
                         NEGATIF si score <= -0.15 (était -0.30)
    - Fallback note_quanti : si le lexique ne trouve aucun mot ET qu'une note
      chiffrée est fournie, on l'utilise pour inférer le sentiment
    - Retourne toujours exactement 'NEGATIF', 'NEUTRE' ou 'POSITIF'
    """
    tokens = nettoyer(texte).split()
    score_total, nb_mots = 0.0, 0
    intensif = 1.0
    negation = False
    NEGATIONS = {"pas", "ne", "plus", "jamais", "aucun", "rien", "sans", "ni"}

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NEGATIONS:
            negation = True
            i += 1
            continue

        bigram = token + " " + tokens[i + 1] if i + 1 < len(tokens) else ""
        mot = bigram if bigram in LEXIQUE else token

        if mot in LEXIQUE:
            entree = LEXIQUE[mot]
            if entree["type"] == "intensificateur":
                intensif = entree["score"]
                if mot == bigram:
                    i += 1
            else:
                s = entree["score"] * intensif
                if negation:
                    s = -s * 0.8
                score_total += s
                nb_mots += 1
                intensif, negation = 1.0, False
        i += 1

    # ── Fallback note_quanti si lexique vide ─────────────────────────────────
    if nb_mots == 0:
        if note_quanti and 1 <= note_quanti <= 5:
            return _sentiment_depuis_note(note_quanti)
        return 0.0, "NEUTRE"

    score = max(-1.0, min(1.0, score_total / nb_mots))

    # CORRECTIF : seuils resserrés ±0.15 au lieu de ±0.30
    if score <= SEUIL_NEGATIF:
        label = "NEGATIF"
    elif score >= SEUIL_POSITIF:
        label = "POSITIF"
    else:
        label = "NEUTRE"

    return round(score, 4), label

# =============================================================================
# DÉTECTION BURNOUT
# =============================================================================

def detecter_burnout(texte: str, score_sentiment: float) -> bool:
    texte_low = texte.lower()
    return any(t in texte_low for t in BURNOUT_TRIGGERS) and score_sentiment < -0.55

# =============================================================================
# GESTION MOTEUR
# =============================================================================

def set_active_model(model_type: str):
    if model_type not in MODELS_DISPONIBLES:
        raise ValueError(f"model_type doit être : {MODELS_DISPONIBLES}")
    _ecrire_modele_persistant(model_type)
    print(f"🔄 Moteur → {model_type}")


def get_active_model() -> str:
    return _lire_modele_persistant()


def get_model_info() -> dict:
    return {
        "active_model":        _lire_modele_persistant(),
        "available_models":    MODELS_DISPONIBLES,
        "zeroshot_loaded":     _model_zeroshot is not None,
        "tfidf_lr_loaded":     _model_lr is not None,
        "tfidf_nb_loaded":     _model_nb is not None,
        "zeroshot_themes":     list(THEMES_ZEROSHOT.keys()),
        "zeroshot_model_name": ZEROSHOT_MODEL_NAME,
        "seuil_positif":       SEUIL_POSITIF,
        "seuil_negatif":       SEUIL_NEGATIF,
    }

# =============================================================================
# ANALYSE PRINCIPALE
# =============================================================================

def analyser(texte: str, note_quanti: int = None) -> dict:
    """
    Analyse complète d'un verbatim.
    note_quanti (optionnel) : note 1-5 pour fallback sentiment si lexique insuffisant.
    Retourne : {theme, confiance, sentiment, score_sentiment, signal_burnout, moteur}
    """
    moteur = _lire_modele_persistant()

    # ── Classification thème
    if moteur == "zeroshot":
        try:
            theme, confiance = _classifier_theme_zeroshot(texte)
        except Exception as e:
            print(f"⚠️  Zero-shot indisponible ({e}) → logistic")
            _ecrire_modele_persistant("logistic")
            theme, confiance = _classifier_theme_tfidf(texte, "logistic")
    else:
        theme, confiance = _classifier_theme_tfidf(texte, moteur)

    # ── Sentiment avec fallback note
    score_sent, label_sent = calculer_sentiment(texte, note_quanti)

    # ── Burnout
    signal_burnout = detecter_burnout(texte, score_sent)

    # ── Log debug (visible dans la console Flask)
    print(f"📝 ANALYSE | thème={theme:20} conf={confiance:.2f} | "
          f"sentiment={label_sent:8} score={score_sent:+.3f} | "
          f"burnout={'OUI' if signal_burnout else 'non':3} | "
          f"moteur={moteur} | texte='{texte[:60]}...'")

    return {
        "theme":           theme,
        "confiance":       confiance,
        "sentiment":       label_sent,    # toujours 'NEGATIF'|'NEUTRE'|'POSITIF'
        "score_sentiment": score_sent,
        "signal_burnout":  signal_burnout,
        "moteur":          moteur,
    }

# =============================================================================
# RECOMMANDATIONS
# =============================================================================

def charger_recommandations() -> dict:
    with open(RECOS_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_recommandation(theme: str, pct_negatif: float) -> Optional[dict]:
    recos = charger_recommandations()
    if theme not in recos:
        return None
    reco = recos[theme]
    if pct_negatif >= reco["seuil"]:
        return reco
    return None


# =============================================================================
# SCRIPT STANDALONE — test rapide
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  QVT NLP Engine v3 — Test sentiment avec nouveaux seuils ±0.15")
    print("=" * 70)

    tests = [
        ("super équipe, vraiment soudée et entraide au quotidien", 5),
        ("bonne ambiance en général", 4),
        ("ça se passe correctement", 3),
        ("quelques tensions mais gérable", 2),
        ("je suis complètement débordé, travaille jusqu'à 21h", 1),
        ("mon manager ne donne jamais de retour, très frustrant", 1),
        ("les outils plantent constamment", 2),
        ("je n'arrive plus à dormir tellement je suis à bout", 1),
    ]

    entrainer_modele(model_type="logistic")
    set_active_model("logistic")

    print(f"\n{'Texte':50} {'Note':5} {'Thème':20} {'Sentiment':10} {'Score':7}")
    print("-" * 100)
    for texte, note in tests:
        r = analyser(texte, note)
        burnout = " 🚨" if r["signal_burnout"] else ""
        print(f"  {texte[:48]:48} {note:5}  {r['theme']:20} {r['sentiment']:10} {r['score_sentiment']:+.3f}{burnout}")