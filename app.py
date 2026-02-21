"""
app.py
------
Application Flask principale — QVT Agent.

Routes :
  GET  /                        → Formulaire collaborateur
  POST /soumettre               → Traitement NLP + sauvegarde
  GET  /merci                   → Page de confirmation
  GET  /dashboard               → Dashboard RH
  GET  /api/stats               → API JSON pour Chart.js
  POST /api/model/set/<type>    → Change le moteur actif
  GET  /api/model/current       → Info sur le moteur actif
  GET  /api/model/zeroshot/themes        → Liste les thèmes zero-shot
  POST /api/model/zeroshot/themes/add   → Ajoute un thème zero-shot à chaud

Moteurs disponibles : "logistic" | "naive_bayes" | "zeroshot"
Pour changer le moteur par défaut, modifiez DEFAULT_MODEL ci-dessous.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
import json, os

from nlp_engine import (
    analyser,
    entrainer_modele,
    get_recommandation,
    charger_recommandations,
    set_active_model,
    get_active_model,
    get_model_info,
    ajouter_theme_zeroshot,
    lister_themes_zeroshot,
    THEMES_ZEROSHOT,
    MODELS_DISPONIBLES,
)
from database import (
    sauvegarder_retour,
    get_stats_themes,
    get_tendance,
    get_score_global,
    get_alerte_burnout,
    get_mois_disponibles,
)

app = Flask(__name__)

# ── Moteur par défaut au démarrage ────────────────────────────────────────────
# Changez ici pour démarrer directement en mode zero-shot :
#   DEFAULT_MODEL = "zeroshot"
DEFAULT_MODEL = "logistic"

# =============================================================================
# INITIALISATION AU DÉMARRAGE
# =============================================================================

print("🚀 Démarrage QVT Agent...")

# Toujours entraîner les modèles TF-IDF si nécessaire
entrainer_modele(model_type="all")
print("✅ Modèles TF-IDF prêts (Logistic Regression + Naive Bayes).")

# N'écrase le moteur persisté QUE si aucun fichier n'existe encore.
# Ainsi, quand Flask recharge le processus (debug=True), il conserve
# le moteur choisi par l'utilisateur via l'API plutôt que de repasser
# systématiquement au DEFAULT_MODEL.
from nlp_engine import _ACTIVE_MODEL_FILE as _AMF
import os as _os
if not _os.path.exists(_AMF):
    set_active_model(DEFAULT_MODEL)
    print(f"   (premier démarrage → moteur par défaut : {DEFAULT_MODEL})")
else:
    print(f"   (moteur persisté conservé : {get_active_model()})")

if get_active_model() == "zeroshot":
    print("ℹ️  Mode zero-shot actif — modèle CamemBERT chargé au premier appel.")

RECOS = charger_recommandations()
print(f"🔧 Moteur actif : {get_active_model()}\n")

# =============================================================================
# ROUTE 1 : Formulaire collaborateur
# =============================================================================

@app.route("/")
def formulaire():
    return render_template("formulaire.html")

# =============================================================================
# ROUTE 2 : Traitement de la soumission
# =============================================================================

@app.route("/soumettre", methods=["POST"])
def soumettre():
    texte    = request.form.get("verbatim", "").strip()
    note_str = request.form.get("note", "")

    if len(texte) < 10:
        return redirect(url_for("formulaire"))

    note_quanti = int(note_str) if note_str.isdigit() and 1 <= int(note_str) <= 5 else None

    # ── Analyse NLP (le texte brut n'est jamais stocké)
    resultat = analyser(texte)

    # ── Sauvegarde anonymisée en base
    sauvegarder_retour(resultat, note_quanti)

    return redirect(url_for("merci"))

# =============================================================================
# ROUTE 3 : Page de confirmation
# =============================================================================

@app.route("/merci")
def merci():
    return render_template("merci.html")

# =============================================================================
# ROUTE 4 : Dashboard RH
# =============================================================================

@app.route("/dashboard")
def dashboard():
    mois_dispo = get_mois_disponibles()
    mois_selec = request.args.get("mois", mois_dispo[0] if mois_dispo else None)

    stats_themes    = get_stats_themes(mois_selec)
    score_global    = get_score_global(mois_selec)
    alerte_burnout  = get_alerte_burnout(mois_selec)
    tendances       = get_tendance()

    # ── Recommandations déclenchées selon les seuils configurés
    recommandations_actives = []
    for stat in stats_themes:
        pct  = stat["pct_negatif"] / 100
        reco = get_recommandation(stat["theme"], pct)
        if reco:
            recommandations_actives.append({
                "theme":          stat["theme"],
                "label":          reco["label"],
                "emoji":          reco["emoji"],
                "color":          reco["color"],
                "pct_negatif":    stat["pct_negatif"],
                "action_directe": reco["action_directe"],
                "actions":        reco["actions"],
                "kpi":            reco["kpi"],
                "risque":         reco["risque"],
            })

    # ── Enrichissement visuel des stats thèmes
    for stat in stats_themes:
        info = RECOS.get(stat["theme"], {})
        stat["label"] = info.get("label", stat["theme"])
        stat["emoji"] = info.get("emoji", "📊")
        stat["color"] = info.get("color", "#888888")
        pct = stat["pct_negatif"]
        if pct >= 70:
            stat["urgence"] = "critique"; stat["urgence_label"] = "🔴 CRITIQUE"
        elif pct >= 60:
            stat["urgence"] = "haute";    stat["urgence_label"] = "🔴 HAUTE"
        elif pct >= 40:
            stat["urgence"] = "vigilance"; stat["urgence_label"] = "🟡 VIGILANCE"
        else:
            stat["urgence"] = "ok";       stat["urgence_label"] = "🟢 OK"

    # ── Données Chart.js pour le graphique de tendance
    chart_tendances = {}
    for theme, points in tendances.items():
        info = RECOS.get(theme, {})
        chart_tendances[theme] = {
            "label":  info.get("label", theme),
            "color":  info.get("color", "#888888"),
            "points": points,
        }

    # ── Info moteur actif (affichée dans le dashboard)
    model_info = get_model_info()

    return render_template("dashboard.html",
        stats_themes=stats_themes,
        score_global=score_global,
        alerte_burnout=alerte_burnout,
        recommandations=recommandations_actives,
        chart_tendances=json.dumps(chart_tendances),
        mois_dispo=mois_dispo,
        mois_selec=mois_selec,
        model_info=model_info,          # ← nouveau : affiché dans le dashboard
    )

# =============================================================================
# ROUTE 5 : API JSON
# =============================================================================

@app.route("/api/stats")
def api_stats():
    mois = request.args.get("mois")
    return jsonify({
        "themes":       get_stats_themes(mois),
        "score_global": get_score_global(mois),
        "burnout":      get_alerte_burnout(mois),
        "tendances":    get_tendance(),
        "moteur_actif": get_active_model(),
    })

# =============================================================================
# ROUTES 6 : Gestion des modèles (admin)
# =============================================================================

@app.route("/api/model/set/<model_type>", methods=["POST"])
def set_model(model_type):
    """
    Change le moteur de classification actif.
    Valeurs acceptées : logistic | naive_bayes | zeroshot

    Pour zeroshot : le modèle CamemBERT-XNLI sera chargé au premier appel
    d'analyser() (téléchargement ~440 MB si absent du cache).
    """
    if model_type not in MODELS_DISPONIBLES:
        return jsonify({
            "error": f"Moteur invalide. Valeurs acceptées : {MODELS_DISPONIBLES}"
        }), 400

    set_active_model(model_type)
    return jsonify({
        "status":       "success",
        "active_model": get_active_model(),
        "message":      f"Moteur changé → {model_type}",
    })


@app.route("/api/model/current")
def get_model_current():
    """Retourne l'état complet des modèles."""
    return jsonify(get_model_info())


@app.route("/api/model/zeroshot/themes")
def get_zeroshot_themes():
    """Liste les thèmes actuellement configurés pour le moteur zero-shot."""
    return jsonify({
        "themes":      THEMES_ZEROSHOT,
        "model_name":  "BaptisteDoyen/camembert-base-xnli",
        "note":        "Modifiables à chaud via POST /api/model/zeroshot/themes/add"
    })


@app.route("/api/model/zeroshot/themes/add", methods=["POST"])
def add_zeroshot_theme():
    """
    Ajoute ou met à jour un thème zero-shot à chaud.
    Body JSON : {"cle": "MON_THEME", "label": "description en français naturel"}
    Aucun réentraînement nécessaire.
    """
    data = request.get_json()
    cle   = data.get("cle", "").strip().upper()
    label = data.get("label", "").strip()

    if not cle or not label:
        return jsonify({"error": "Les champs 'cle' et 'label' sont requis."}), 400

    ajouter_theme_zeroshot(cle, label)
    return jsonify({
        "status":  "success",
        "theme":   {cle: label},
        "message": f"Thème '{cle}' ajouté/mis à jour.",
    })


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)