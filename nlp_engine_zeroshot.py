"""
nlp_engine.py
-------------
Moteur NLP du projet QVT — version Zero-Shot (sans données d'entraînement).
Responsabilités :
  1. Classifier les thèmes via zero-shot (CamemBERT-XNLI)
  2. Calculer le score de sentiment (lexique + règles, inchangé)
  3. Détecter les signaux faibles burnout
  4. Exposer une fonction analyser(texte) → dict résultat

Modèle : BaptisteDoyen/camembert-base-xnli
  → CamemBERT fine-tuné sur XNLI, spécialisé français, ~440 MB.
  → Téléchargé automatiquement depuis HuggingFace au premier appel.
  → Mis en cache localement dans models/zeroshot_cache/

Avantages vs TF-IDF + LogReg :
  - Aucun fichier d'entraînement nécessaire
  - Thèmes modifiables à la volée (juste changer THEMES_LABELS)
  - Meilleure généralisation sur des formulations inattendues
  - Compréhension contextuelle réelle (BERT)

Inconvénients :
  - Plus lent (~0.5–2s par texte sans GPU, ~0.05s avec GPU)
  - Requiert ~440 MB de mémoire / disque
  - Nécessite : pip install transformers torch sentencepiece
"""

import os, re, csv, json
import pandas as pd
from typing import Optional

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CACHE_DIR  = os.path.join(MODEL_DIR, "zeroshot_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

LEXIQUE_CSV  = os.path.join(DATA_DIR, "lexique_sentiment.csv")
RECOS_JSON   = os.path.join(DATA_DIR, "recommandations.json")

# ── Modèle zero-shot ──────────────────────────────────────────────────────────
# CamemBERT fine-tuné sur XNLI — le meilleur choix pour le français
ZEROSHOT_MODEL_NAME = "BaptisteDoyen/camembert-base-xnli"

# Alternative si vous préférez un modèle multilingue plus léger :
# ZEROSHOT_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"  # ~120 MB, bon en FR

# ── Thèmes QVT — modifiables librement sans réentraînement ───────────────────
# Format : { "clé_interne": "label en français naturel pour le modèle" }
# Le modèle compare le texte à chaque hypothèse "Ce texte parle de {label}."
THEMES_LABELS: dict[str, str] = {
    "CHARGE_TRAVAIL":    "surcharge de travail, heures supplémentaires ou épuisement professionnel",
    "MANAGEMENT":        "management, relations avec le responsable ou le manager",
    "COMMUNICATION":     "communication interne, échanges d'informations ou transparence",
    "AMBIANCE":          "ambiance d'équipe, relations entre collègues ou cohésion",
    "OUTILS":            "outils informatiques, logiciels ou équipements de travail",
    "REMUNERATION":      "salaire, rémunération, augmentation ou avantages financiers",
    "EVOLUTION":         "évolution de carrière, formation professionnelle ou perspectives",
    "EQUILIBRE_VIE":     "équilibre entre vie professionnelle et vie personnelle",
    "RECONNAISSANCE":    "reconnaissance au travail, valorisation ou feedback positif",
    "CONDITIONS":        "conditions de travail, locaux, bruit ou ergonomie",
}

# Seuil de confiance : en dessous → NON_CLASSE
SEUIL_CONFIANCE = 0.20

# ── Mots déclencheurs burnout (inchangé) ─────────────────────────────────────
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

# ── 2. PIPELINE ZERO-SHOT ─────────────────────────────────────────────────────
_classifier = None  # cache en mémoire

def charger_classifier():
    """
    Charge le pipeline zero-shot au premier appel.
    Téléchargement automatique depuis HuggingFace (~440 MB, une seule fois).
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Le module 'transformers' est requis.\n"
            "Installez-le avec : pip install transformers torch sentencepiece"
        )

    print(f"⏳ Chargement du modèle zero-shot : {ZEROSHOT_MODEL_NAME}")
    print(f"   (Téléchargement au premier appel ~440 MB, puis mise en cache dans {CACHE_DIR})")

    _classifier = pipeline(
        task="zero-shot-classification",
        model=ZEROSHOT_MODEL_NAME,
        cache_dir=CACHE_DIR,
        # device=0,  # Décommentez pour utiliser le GPU (CUDA)
    )
    print("✅ Modèle chargé.")
    return _classifier

def classifier_theme(texte: str) -> tuple[str, float]:
    """
    Classifie le thème du texte via zero-shot NLI.

    Retourne (theme_clé, score_confiance).
    Le modèle teste chaque label comme hypothèse NLI :
    "Ce texte parle de {label}." → score de probabilité d'entailment.

    Note sur hypothesis_template :
      CamemBERT-XNLI ayant été entraîné en français, on utilise
      un template français pour de meilleures performances.
    """
    clf = charger_classifier()

    # Labels en français naturel (valeurs du dict THEMES_LABELS)
    labels_list = list(THEMES_LABELS.values())
    cles_list   = list(THEMES_LABELS.keys())

    result = clf(
        sequences=texte,
        candidate_labels=labels_list,
        hypothesis_template="Ce texte parle de {}.",
        multi_label=False,   # On veut UN seul thème dominant
    )
    # result["labels"][0] est le label avec le score le plus élevé
    best_label = result["labels"][0]
    best_score = result["scores"][0]

    # Retrouver la clé interne correspondante
    idx = labels_list.index(best_label)
    theme_key = cles_list[idx]

    if best_score < SEUIL_CONFIANCE:
        theme_key = "NON_CLASSE"

    return theme_key, round(float(best_score), 4)

# ── 3. LEXIQUE SENTIMENT (identique à l'original) ────────────────────────────
def charger_lexique() -> dict:
    """Retourne {mot: (score, type)} depuis lexique_sentiment.csv"""
    lexique = {}
    if not os.path.exists(LEXIQUE_CSV):
        print(f"⚠️  Lexique non trouvé : {LEXIQUE_CSV} — sentiment désactivé.")
        return lexique
    with open(LEXIQUE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lexique[row["mot"].lower().strip()] = {
                "score": float(row["score"]),
                "type":  row["type"]
            }
    return lexique

LEXIQUE = charger_lexique()

def calculer_sentiment(texte: str) -> tuple[float, str]:
    """
    Calcule le score de sentiment d'un texte via lexique + règles.
    Applique les intensificateurs sur le mot suivant.
    Gère la négation simple (ne/pas/jamais/aucun…).
    Retourne (score_float, label_string).
    """
    tokens = nettoyer(texte).split()
    score_total = 0.0
    nb_mots_sentiments = 0
    intensificateur_actif = 1.0
    negation_active = False

    NEGATIONS = {"pas", "ne", "plus", "jamais", "aucun", "rien", "sans", "ni"}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in NEGATIONS:
            negation_active = True
            i += 1
            continue

        bigram = token + " " + tokens[i + 1] if i + 1 < len(tokens) else ""
        mot_a_tester = bigram if bigram in LEXIQUE else token

        if mot_a_tester in LEXIQUE:
            entree = LEXIQUE[mot_a_tester]

            if entree["type"] == "intensificateur":
                intensificateur_actif = entree["score"]
                if mot_a_tester == bigram:
                    i += 1
            else:
                s = entree["score"] * intensificateur_actif
                if negation_active:
                    s = -s * 0.8
                score_total += s
                nb_mots_sentiments += 1
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

# ── 4. DÉTECTION SIGNAL FAIBLE BURNOUT (identique) ───────────────────────────
def detecter_burnout(texte: str, score_sentiment: float) -> bool:
    """
    Retourne True si le texte contient des signaux de burnout.
    Critères : mot déclencheur présent ET score sentiment < -0.55
    """
    texte_low = texte.lower()
    trigger_trouve = any(t in texte_low for t in BURNOUT_TRIGGERS)
    return trigger_trouve and score_sentiment < -0.55

# ── 5. FONCTION PRINCIPALE : analyse complète d'un verbatim ──────────────────
def analyser(texte: str) -> dict:
    """
    Point d'entrée principal.
    Reçoit un texte brut, retourne un dict avec thème + sentiment + score + burnout.
    Le texte brut N'EST PAS retourné dans le résultat (anonymat).
    """
    # ── Détection du thème (zero-shot)
    theme_predit, confiance = classifier_theme(texte)

    # ── Sentiment (lexique)
    score_sent, label_sent = calculer_sentiment(texte)

    # ── Signal burnout
    signal_burnout = detecter_burnout(texte, score_sent)

    return {
        "theme":           theme_predit,
        "confiance":       confiance,
        "sentiment":       label_sent,
        "score_sentiment": score_sent,
        "signal_burnout":  signal_burnout,
    }

# ── 6. RECOMMANDATIONS (identique) ───────────────────────────────────────────
def charger_recommandations() -> dict:
    with open(RECOS_JSON, encoding="utf-8") as f:
        return json.load(f)

def get_recommandation(theme: str, pct_negatif: float) -> Optional[dict]:
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

# ── 7. UTILITAIRE : mise à jour des thèmes à chaud ───────────────────────────
def ajouter_theme(cle: str, label_fr: str):
    """
    Ajoute ou met à jour un thème sans réentraînement.
    Exemple : ajouter_theme("INCLUSION", "diversité, inclusion et égalité au travail")
    """
    THEMES_LABELS[cle] = label_fr
    print(f"✅ Thème ajouté/mis à jour : {cle} → \"{label_fr}\"")

def lister_themes():
    """Affiche les thèmes configurés."""
    print("\n📋 Thèmes QVT configurés :")
    for cle, label in THEMES_LABELS.items():
        print(f"  {cle:20} → {label}")

# ── Script standalone : test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    lister_themes()
    print("\n🧪 Tests zero-shot (CamemBERT-XNLI) :")
    print("   (Le modèle sera téléchargé au premier appel si absent)\n")

    tests = [
        "Je suis complètement débordé, je travaille jusqu'à 21h tous les soirs.",
        "Mon manager ne nous donne jamais de retour, c'est très frustrant.",
        "Les outils plantent constamment, je perds un temps fou.",
        "Super ambiance dans l'équipe, on s'entraide vraiment bien.",
        "Je n'arrive plus à dormir tellement je pense au travail.",
        "On m'a promis une augmentation et elle n'est jamais arrivée.",
        "Je voudrais pouvoir télétravailler plus souvent pour mieux gérer ma famille.",
        "Personne ne reconnaît nos efforts, c'est démotivant.",
    ]

    for t in tests:
        r = analyser(t)
        burnout = "🚨 BURNOUT" if r["signal_burnout"] else ""
        print(f"  [{r['theme']:20}] conf={r['confiance']:.2f} [{r['sentiment']:8}] score={r['score_sentiment']:+.2f}  {burnout}")
        print(f"   → {t[:80]}")
        print()
