"""
nlp_engine.py
-------------
Moteur NLP unifié du projet QVT — 3 moteurs de classification disponibles :

  1. "logistic"    → TF-IDF + Logistic Regression  (rapide, léger, nécessite données d'entraînement)
  2. "naive_bayes" → TF-IDF + Naive Bayes          (très rapide, léger, nécessite données d'entraînement)
  3. "zeroshot"    → CamemBERT-XNLI zero-shot      (lent ~1-2s/texte, 440 MB, aucune donnée requise)

Changement de moteur :
  • Via API  : POST /api/model/set/zeroshot
  • Via code : set_active_model("zeroshot")
  • Au démarrage dans app.py : MODEL_TYPE = "zeroshot"

Le sentiment et la détection burnout restent identiques (lexique + règles)
quel que soit le moteur de classification choisi.
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

# ── Moteurs disponibles ───────────────────────────────────────────────────────
MODELS_DISPONIBLES = ["logistic", "naive_bayes", "zeroshot"]

# Fichier de persistance du moteur actif.
# Permet de survivre aux redémarrages du reloader Flask (debug=True).
# Le reloader Flask tue et recrée le processus à chaque changement de code,
# ce qui réinitialise toutes les variables globales en mémoire.
# En lisant/écrivant dans un fichier, le choix de moteur survit à ces redémarrages.
_ACTIVE_MODEL_FILE = os.path.join(MODEL_DIR, "active_model.txt")

def _lire_modele_persistant() -> str:
    """Lit le moteur actif depuis le fichier de persistance (ou retourne le défaut)."""
    try:
        with open(_ACTIVE_MODEL_FILE, "r") as f:
            val = f.read().strip()
            if val in MODELS_DISPONIBLES:
                return val
    except FileNotFoundError:
        pass
    return "logistic"

def _ecrire_modele_persistant(model_type: str):
    """Écrit le moteur actif dans le fichier de persistance."""
    os.makedirs(os.path.dirname(_ACTIVE_MODEL_FILE), exist_ok=True)
    with open(_ACTIVE_MODEL_FILE, "w") as f:
        f.write(model_type)

# ── Caches en mémoire ─────────────────────────────────────────────────────────
_model_lr        = None   # Pipeline scikit-learn LR
_model_nb        = None   # Pipeline scikit-learn NB
_model_zeroshot  = None   # Pipeline HuggingFace zero-shot

# ── Configuration zero-shot ───────────────────────────────────────────────────
# Modèle multilingue robuste, compatible toutes versions de transformers récentes.
# Remplace BaptisteDoyen/camembert-base-xnli (incompatible avec transformers >= 4.36)
ZEROSHOT_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

# Fallback si le modèle principal échoue
ZEROSHOT_MODEL_FALLBACK = "cross-encoder/nli-MiniLM2-L6-H768"

# Thèmes QVT pour le zero-shot : clé interne → description naturelle en français
# Ces labels sont directement compris par le modèle, modifiables sans réentraînement
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

SEUIL_CONFIANCE_ZEROSHOT = 0.20  # En dessous → NON_CLASSE
SEUIL_CONFIANCE_TFIDF    = 0.15  # En dessous → NON_CLASSE

# ── Mots déclencheurs burnout ─────────────────────────────────────────────────
BURNOUT_TRIGGERS = [
    "burnout", "craquer", "plus de force", "à bout", "plus envie",
    "ras-le-bol", "démissionner", "je veux partir", "je n'en peux plus",
    "vidé", "épuisement total", "plus capable", "arrêt maladie",
    "ne dors plus", "n'arrive plus à dormir"
]

# =============================================================================
# 1. NETTOYAGE
# =============================================================================

def nettoyer(texte: str) -> str:
    """Minuscules + suppression ponctuation lourde, garde les apostrophes."""
    texte = texte.lower().strip()
    texte = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüç]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte

# =============================================================================
# 2. MODÈLES TF-IDF (LOGISTIC REGRESSION & NAIVE BAYES)
# =============================================================================

def entrainer_modele(force=False, model_type="all"):
    """
    Entraîne les modèles TF-IDF + classifieurs à partir de verbatims_entrainement.csv.
    model_type : "logistic" | "naive_bayes" | "all"
    Ne ré-entraîne pas si les fichiers existent déjà (sauf force=True).
    N'a aucun effet sur le moteur zero-shot (pas de données d'entraînement requises).
    """
    # Le zero-shot ne nécessite pas d'entraînement
    if model_type == "zeroshot":
        print("ℹ️  Zero-shot : aucun entraînement nécessaire.")
        return

    if not os.path.exists(VERBATIMS_CSV):
        print(f"⚠️  Fichier d'entraînement introuvable : {VERBATIMS_CSV}")
        print("   Les modèles TF-IDF ne peuvent pas être entraînés.")
        return

    print(f"⏳ Entraînement {'modèles TF-IDF' if model_type == 'all' else model_type}...")
    df = pd.read_csv(VERBATIMS_CSV, encoding="utf-8")
    df = df.dropna(subset=["texte", "theme"])
    df["texte_propre"] = df["texte"].apply(nettoyer)
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
            print("\n📊 Logistic Regression :")
            print(classification_report(y_test, pipe.predict(X_test)))
            with open(MODEL_LR_FILE, "wb") as f:
                pickle.dump(pipe, f)
            print(f"✅ Logistic Regression sauvegardé → {MODEL_LR_FILE}")

    if model_type in ["naive_bayes", "all"]:
        if not os.path.exists(MODEL_NB_FILE) or force:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", MultinomialNB(alpha=1.0))
            ])
            pipe.fit(X_train, y_train)
            print("\n📊 Naive Bayes :")
            print(classification_report(y_test, pipe.predict(X_test)))
            with open(MODEL_NB_FILE, "wb") as f:
                pickle.dump(pipe, f)
            print(f"✅ Naive Bayes sauvegardé → {MODEL_NB_FILE}")


def _charger_modele_tfidf(model_type: str):
    """Charge un modèle TF-IDF depuis le disque (avec entraînement auto si absent)."""
    path = MODEL_NB_FILE if model_type == "naive_bayes" else MODEL_LR_FILE
    if not os.path.exists(path):
        print(f"⚠️  Modèle {model_type} non trouvé → entraînement automatique...")
        entrainer_modele(force=True, model_type=model_type)
    with open(path, "rb") as f:
        return pickle.load(f)


def _classifier_theme_tfidf(texte: str, model_type: str) -> tuple[str, float]:
    """
    Classifie le thème via TF-IDF + classifieur scikit-learn.
    Utilise le cache en mémoire pour éviter de recharger à chaque appel.
    """
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
# 3. MODÈLE ZERO-SHOT (CamemBERT-XNLI)
# =============================================================================

def _charger_zeroshot():
    """
    Charge le pipeline zero-shot HuggingFace au premier appel.
    Téléchargement automatique puis mise en cache locale.
    Essaie d'abord ZEROSHOT_MODEL_NAME, puis ZEROSHOT_MODEL_FALLBACK en cas d'erreur.
    """
    global _model_zeroshot
    if _model_zeroshot is not None:
        return _model_zeroshot

    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Le module 'transformers' est requis pour le moteur zero-shot.\n"
            "Installez-le avec : pip install transformers torch sentencepiece"
        )

    for model_name in [ZEROSHOT_MODEL_NAME, ZEROSHOT_MODEL_FALLBACK]:
        try:
            print(f"⏳ Chargement du modèle zero-shot : {model_name}")
            print(f"   (Cache dans {CACHE_DIR})")
            _model_zeroshot = pipeline(
                task="zero-shot-classification",
                model=model_name,
                cache_dir=CACHE_DIR,
                # device=0,  # Décommenter pour utiliser le GPU CUDA
            )
            print(f"✅ Modèle zero-shot chargé : {model_name}")
            return _model_zeroshot
        except Exception as e:
            print(f"⚠️  Échec chargement {model_name} : {e}")
            print(f"   → Tentative avec le modèle suivant...")

    raise RuntimeError(
        "Impossible de charger un modèle zero-shot compatible.\n"
        "Solutions :\n"
        "  1. Mettre à jour transformers : pip install -U transformers\n"
        "  2. Utiliser le moteur logistic à la place :\n"
        "     curl -X POST http://localhost:5000/api/model/set/logistic\n"
        "  OU supprimez le fichier models/active_model.txt pour revenir au défaut."
    )


def _classifier_theme_zeroshot(texte: str) -> tuple[str, float]:
    """
    Classifie le thème via zero-shot NLI (CamemBERT-XNLI).
    Le modèle évalue chaque label comme hypothèse : "Ce texte parle de {label}."
    Aucune donnée d'entraînement requise — les thèmes sont modifiables à chaud.
    """
    clf = _charger_zeroshot()
    labels_list = list(THEMES_ZEROSHOT.values())
    cles_list   = list(THEMES_ZEROSHOT.keys())

    result = clf(
        sequences=texte,
        candidate_labels=labels_list,
        hypothesis_template="Ce texte parle de {}.",
        multi_label=False,  # Un seul thème dominant
    )

    best_label = result["labels"][0]
    best_score = result["scores"][0]
    idx = labels_list.index(best_label)
    theme_key = cles_list[idx]

    if best_score < SEUIL_CONFIANCE_ZEROSHOT:
        theme_key = "NON_CLASSE"

    return theme_key, round(float(best_score), 4)


def ajouter_theme_zeroshot(cle: str, label_fr: str):
    """
    Ajoute ou modifie un thème zero-shot à chaud, sans réentraînement.
    Exemple : ajouter_theme_zeroshot("INCLUSION", "diversité, inclusion et égalité au travail")
    """
    THEMES_ZEROSHOT[cle] = label_fr
    print(f"✅ Thème zero-shot ajouté/mis à jour : {cle} → \"{label_fr}\"")


def lister_themes_zeroshot():
    """Affiche les thèmes zero-shot configurés."""
    print("\n📋 Thèmes zero-shot configurés :")
    for cle, label in THEMES_ZEROSHOT.items():
        print(f"  {cle:20} → {label}")

# =============================================================================
# 4. SENTIMENT (lexique + règles — identique pour tous les moteurs)
# =============================================================================

def charger_lexique() -> dict:
    """Charge le lexique de sentiment depuis lexique_sentiment.csv."""
    lexique = {}
    if not os.path.exists(LEXIQUE_CSV):
        print(f"⚠️  Lexique introuvable : {LEXIQUE_CSV} — sentiment désactivé.")
        return lexique
    with open(LEXIQUE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lexique[row["mot"].lower().strip()] = {
                "score": float(row["score"]),
                "type":  row["type"]
            }
    return lexique

LEXIQUE = charger_lexique()


def calculer_sentiment(texte: str) -> tuple[float, str]:
    """
    Score de sentiment via lexique + règles linguistiques.
    Gère : intensificateurs, négation simple (pas/ne/jamais/aucun…), bigrammes.
    Retourne (score_float ∈ [-1, 1], label ∈ {POSITIF, NEGATIF, NEUTRE}).
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

    if nb_mots == 0:
        return 0.0, "NEUTRE"

    score = max(-1.0, min(1.0, score_total / nb_mots))
    label = "NEGATIF" if score <= -0.3 else ("POSITIF" if score >= 0.3 else "NEUTRE")
    return round(score, 4), label

# =============================================================================
# 5. DÉTECTION BURNOUT (identique pour tous les moteurs)
# =============================================================================

def detecter_burnout(texte: str, score_sentiment: float) -> bool:
    """
    Retourne True si le texte contient un signal de burnout.
    Condition : mot déclencheur présent ET score sentiment < -0.55.
    """
    texte_low = texte.lower()
    return any(t in texte_low for t in BURNOUT_TRIGGERS) and score_sentiment < -0.55

# =============================================================================
# 6. GESTION DU MODÈLE ACTIF
# =============================================================================

def set_active_model(model_type: str):
    """
    Change le moteur de classification actif.
    Valeurs acceptées : "logistic" | "naive_bayes" | "zeroshot"

    Le changement est persisté dans models/active_model.txt pour survivre
    aux redémarrages du reloader Flask (debug=True).
    """
    if model_type not in MODELS_DISPONIBLES:
        raise ValueError(f"model_type doit être l'un de : {MODELS_DISPONIBLES}")
    _ecrire_modele_persistant(model_type)
    print(f"🔄 Moteur actif changé → {model_type}")


def get_active_model() -> str:
    """Retourne le nom du moteur de classification actuellement actif (lu depuis le fichier)."""
    return _lire_modele_persistant()


def get_model_info() -> dict:
    """Retourne un dict complet sur l'état des modèles."""
    return {
        "active_model":       _lire_modele_persistant(),
        "available_models":   MODELS_DISPONIBLES,
        "zeroshot_loaded":    _model_zeroshot is not None,
        "tfidf_lr_loaded":    _model_lr is not None,
        "tfidf_nb_loaded":    _model_nb is not None,
        "zeroshot_themes":    list(THEMES_ZEROSHOT.keys()),
        "zeroshot_model_name": ZEROSHOT_MODEL_NAME,
    }

# =============================================================================
# 7. ANALYSE PRINCIPALE (point d'entrée unique)
# =============================================================================

def analyser(texte: str) -> dict:
    """
    Analyse complète d'un verbatim collaborateur.

    Entrée  : texte brut (non stocké après analyse)
    Sortie  : {theme, confiance, sentiment, score_sentiment, signal_burnout, moteur}

    Le moteur utilisé dépend de _active_model_type (modifiable via set_active_model()).
    """
    moteur = _lire_modele_persistant()  # toujours lire depuis le fichier (survit au reloader)

    # ── Classification du thème selon le moteur actif
    if moteur == "zeroshot":
        try:
            theme, confiance = _classifier_theme_zeroshot(texte)
        except Exception as e:
            print(f"⚠️  Moteur zero-shot indisponible ({e})")
            print("   → Bascule automatique sur 'logistic'")
            _ecrire_modele_persistant("logistic")   # corrige le fichier persistant
            theme, confiance = _classifier_theme_tfidf(texte, "logistic")
    else:
        theme, confiance = _classifier_theme_tfidf(texte, moteur)

    # ── Sentiment (identique pour tous les moteurs)
    score_sent, label_sent = calculer_sentiment(texte)

    # ── Détection burnout
    signal_burnout = detecter_burnout(texte, score_sent)

    return {
        "theme":           theme,
        "confiance":       confiance,
        "sentiment":       label_sent,
        "score_sentiment": score_sent,
        "signal_burnout":  signal_burnout,
        "moteur":          moteur,          # ← nouveau : traçabilité du moteur utilisé
    }

# =============================================================================
# 8. RECOMMANDATIONS
# =============================================================================

def charger_recommandations() -> dict:
    """Charge le fichier recommandations.json."""
    with open(RECOS_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_recommandation(theme: str, pct_negatif: float) -> Optional[dict]:
    """
    Retourne la recommandation pour un thème si pct_negatif >= seuil configuré.
    Retourne None si le thème est inconnu ou si le seuil n'est pas atteint.
    """
    recos = charger_recommandations()
    if theme not in recos:
        return None
    reco = recos[theme]
    if pct_negatif >= reco["seuil"]:
        return reco
    return None

# =============================================================================
# SCRIPT STANDALONE — test rapide des 3 moteurs
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  QVT NLP Engine — Test des 3 moteurs")
    print("=" * 65)

    TESTS = [
        "Je suis complètement débordé, je travaille jusqu'à 21h tous les soirs.",
        "Mon manager ne nous donne jamais de retour, c'est très frustrant.",
        "Les outils plantent constamment, je perds un temps fou.",
        "Super ambiance dans l'équipe, on s'entraide vraiment bien.",
        "Je n'arrive plus à dormir tellement je pense au travail.",
        "On m'a promis une augmentation et elle n'est jamais arrivée.",
    ]

    for moteur in ["logistic", "naive_bayes", "zeroshot"]:
        print(f"\n{'─'*65}")
        print(f"  🤖 Moteur : {moteur.upper()}")
        print(f"{'─'*65}")

        if moteur in ["logistic", "naive_bayes"]:
            entrainer_modele(model_type=moteur)

        set_active_model(moteur)

        for t in TESTS:
            try:
                r = analyser(t)
                burnout = " 🚨 BURNOUT" if r["signal_burnout"] else ""
                print(f"  [{r['theme']:20}] conf={r['confiance']:.2f} "
                      f"[{r['sentiment']:8}] score={r['score_sentiment']:+.2f}{burnout}")
                print(f"   → {t[:75]}")
                print()
            except Exception as e:
                print(f"  ⚠️  Erreur avec le moteur {moteur} : {e}")
                break