"""
app.py
------
Application Flask principale.
Routes :
  GET  /            → Formulaire collaborateur
  POST /soumettre   → Traitement NLP + sauvegarde
  GET  /merci       → Page de confirmation
  GET  /dashboard   → Dashboard RH
  GET  /api/stats   → API JSON pour Chart.js
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
import json, os

from nlp_engine import analyser, entrainer_modele, get_recommandation, charger_recommandations
from database  import (sauvegarder_retour, get_stats_themes, get_tendance,
                       get_score_global, get_alerte_burnout, get_mois_disponibles)

app = Flask(__name__)

# ── Entraîne le modèle au démarrage (une seule fois) ────────────────────────
print("🚀 Démarrage QVT Agent...")
entrainer_modele()
print("✅ Modèle prêt.\n")

RECOS = charger_recommandations()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE 1 : Formulaire collaborateur
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def formulaire():
    return render_template("formulaire.html")

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE 2 : Traitement de la soumission
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/soumettre", methods=["POST"])
def soumettre():
    texte = request.form.get("verbatim", "").strip()
    note_str = request.form.get("note", "")

    # Validation basique
    if len(texte) < 10:
        return redirect(url_for("formulaire"))

    note_quanti = int(note_str) if note_str.isdigit() and 1 <= int(note_str) <= 5 else None

    # ── Analyse NLP (le texte n'est jamais stocké après cette ligne)
    resultat = analyser(texte)

    # ── Sauvegarde anonymisée
    sauvegarder_retour(resultat, note_quanti)

    # ── Redirection vers page de remerciement
    return redirect(url_for("merci"))

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE 3 : Page de confirmation
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/merci")
def merci():
    return render_template("merci.html")

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE 4 : Dashboard RH
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    mois_dispo = get_mois_disponibles()
    mois_selec = request.args.get("mois", mois_dispo[0] if mois_dispo else None)

    stats_themes = get_stats_themes(mois_selec)
    score_global = get_score_global(mois_selec)
    alerte_burnout = get_alerte_burnout(mois_selec)
    tendances = get_tendance()

    # ── Calcul des recommandations déclenchées
    recommandations_actives = []
    for stat in stats_themes:
        pct = stat["pct_negatif"] / 100
        reco = get_recommandation(stat["theme"], pct)
        if reco:
            recommandations_actives.append({
                "theme":        stat["theme"],
                "label":        reco["label"],
                "emoji":        reco["emoji"],
                "color":        reco["color"],
                "pct_negatif":  stat["pct_negatif"],
                "action_directe": reco["action_directe"],
                "actions":      reco["actions"],
                "kpi":          reco["kpi"],
                "risque":       reco["risque"],
            })

    # ── Enrichissement des stats avec infos visuelles
    for stat in stats_themes:
        theme_info = RECOS.get(stat["theme"], {})
        stat["label"]  = theme_info.get("label", stat["theme"])
        stat["emoji"]  = theme_info.get("emoji", "📊")
        stat["color"]  = theme_info.get("color", "#888888")
        # Niveau d'urgence visuel
        pct = stat["pct_negatif"]
        if pct >= 70:
            stat["urgence"] = "critique"
            stat["urgence_label"] = "🔴 CRITIQUE"
        elif pct >= 60:
            stat["urgence"] = "haute"
            stat["urgence_label"] = "🔴 HAUTE"
        elif pct >= 40:
            stat["urgence"] = "vigilance"
            stat["urgence_label"] = "🟡 VIGILANCE"
        else:
            stat["urgence"] = "ok"
            stat["urgence_label"] = "🟢 OK"

    # ── Données Chart.js pour le graphique tendance
    chart_tendances = {}
    for theme, points in tendances.items():
        theme_info = RECOS.get(theme, {})
        chart_tendances[theme] = {
            "label":  theme_info.get("label", theme),
            "color":  theme_info.get("color", "#888888"),
            "points": points
        }

    return render_template("dashboard.html",
        stats_themes=stats_themes,
        score_global=score_global,
        alerte_burnout=alerte_burnout,
        recommandations=recommandations_actives,
        chart_tendances=json.dumps(chart_tendances),
        mois_dispo=mois_dispo,
        mois_selec=mois_selec,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE 5 : API JSON (pour requêtes dynamiques)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    mois = request.args.get("mois")
    return jsonify({
        "themes":       get_stats_themes(mois),
        "score_global": get_score_global(mois),
        "burnout":      get_alerte_burnout(mois),
        "tendances":    get_tendance(),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
