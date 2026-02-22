"""
app.py  (version mise à jour — QVT Agent)
-----------------------------------------
Routes :
  GET  /                                   → Formulaire collaborateur
  POST /soumettre                          → Traitement NLP + sauvegarde
  GET  /merci                              → Page de confirmation
  GET  /dashboard                          → Dashboard RH
  GET  /api/stats                          → API JSON pour Chart.js
  POST /api/synthese                       → Synthèse LLM mensuelle ← NOUVEAU
  POST /api/model/set/<type>               → Change le moteur actif
  GET  /api/model/current                  → Info sur le moteur actif
  GET  /api/model/zeroshot/themes          → Liste les thèmes zero-shot
  POST /api/model/zeroshot/themes/add      → Ajoute un thème zero-shot à chaud

Prérequis supplémentaire pour la synthèse :
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
import json, os

# ── Chargement des variables d'environnement depuis .env ────────────────────
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv non installé — utilise les variables système

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

# ── Import Anthropic (optionnel — uniquement pour la synthèse LLM) ────────────
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    print("ℹ️  Module 'anthropic' non installé. Route /api/synthese désactivée.")
    print("   Pour l'activer : pip install anthropic")

app = Flask(__name__)

# ── Moteur par défaut au démarrage ────────────────────────────────────────────
DEFAULT_MODEL = "logistic"

# =============================================================================
# INITIALISATION AU DÉMARRAGE
# =============================================================================

print("🚀 Démarrage QVT Agent...")

entrainer_modele(model_type="all")
print("✅ Modèles TF-IDF prêts (Logistic Regression + Naive Bayes).")

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

    resultat = analyser(texte)
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

    chart_tendances = {}
    for theme, points in tendances.items():
        info = RECOS.get(theme, {})
        chart_tendances[theme] = {
            "label":  info.get("label", theme),
            "color":  info.get("color", "#888888"),
            "points": points,
        }

    model_info = get_model_info()

    return render_template("dashboard.html",
        stats_themes=stats_themes,
        score_global=score_global,
        alerte_burnout=alerte_burnout,
        recommandations=recommandations_actives,
        chart_tendances=json.dumps(chart_tendances),
        mois_dispo=mois_dispo,
        mois_selec=mois_selec,
        model_info=model_info,
    )

# =============================================================================
# ROUTE 5 : API JSON stats
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
# ROUTE 6 : Synthèse mensuelle — générateur local intelligent (100% gratuit)
# =============================================================================

def _generer_synthese_locale(data: dict) -> dict:
    """
    Génère une synthèse narrative RH à partir des stats, sans API externe.
    Analyse les seuils, construit des phrases contextuelles, 100% gratuit.
    """
    mois           = data.get("mois", "N/A")
    score_global   = data.get("score_global", {})
    alerte_burnout = data.get("alerte_burnout", {})
    stats_themes   = data.get("stats_themes", [])

    score      = score_global.get("score_global", 0)
    total      = score_global.get("total", 0)
    nb_signaux = alerte_burnout.get("nb_signaux", 0)
    has_burnout= alerte_burnout.get("alerte", False)

    # ── Qualification du score global
    if score >= 0.3:
        qualif_score = "globalement positif"
        tendance_mot = "favorable"
    elif score >= 0:
        qualif_score = "légèrement positif"
        tendance_mot = "plutôt stable"
    elif score >= -0.3:
        qualif_score = "légèrement négatif"
        tendance_mot = "préoccupant"
    else:
        qualif_score = "négatif"
        tendance_mot = "dégradé"

    # ── Tri des thèmes par criticité
    critiques  = [s for s in stats_themes if s.get("pct_negatif", 0) >= 60]
    vigilance  = [s for s in stats_themes if 40 <= s.get("pct_negatif", 0) < 60]
    positifs   = [s for s in stats_themes if s.get("pct_negatif", 0) < 30
                  and s.get("nb_positif", 0) / max(s.get("total", 1), 1) >= 0.5]

    critiques.sort(key=lambda x: x.get("pct_negatif", 0), reverse=True)
    positifs.sort(key=lambda x: x.get("nb_positif", 0), reverse=True)

    def label(s):
        return s.get("label", s.get("theme", ""))

    # ── RÉSUMÉ EXÉCUTIF
    resume_parts = [
        f"Le mois de {mois} enregistre {total} retours collaborateurs avec un sentiment {qualif_score} "
        f"(score : {score:+.2f}/1.0)."
    ]
    if critiques:
        noms = ", ".join(label(s) for s in critiques[:2])
        resume_parts.append(
            f"{len(critiques)} thème(s) atteignent un taux de négativité critique : {noms}."
        )
    if has_burnout:
        resume_parts.append(
            f"⚠️ {nb_signaux} signal(aux) de détresse ont été détectés ce mois — une attention immédiate est requise."
        )
    if positifs:
        noms_pos = ", ".join(label(s) for s in positifs[:2])
        resume_parts.append(f"Points positifs à noter : {noms_pos} affichent des retours majoritairement satisfaisants.")
    resume = " ".join(resume_parts)

    # ── ALERTES
    alertes_themes = [label(s) for s in critiques[:3]]
    alertes_parts = []
    if critiques:
        top = critiques[0]
        alertes_parts.append(
            f"Le thème \"{label(top)}\" concentre {top.get('pct_negatif', 0)}% de retours négatifs "
            f"sur {top.get('total', 0)} réponses — c'est le point le plus urgent à adresser."
        )
    if len(critiques) > 1:
        alertes_parts.append(
            f"Les thèmes {', '.join(label(s) for s in critiques[1:3])} dépassent également le seuil critique de 60%."
        )
    if has_burnout:
        alertes_parts.append(
            f"La présence de {nb_signaux} signal(aux) de détresse nécessite un dispositif d'écoute confidentiel immédiat."
        )
    if vigilance and not critiques:
        alertes_parts.append(
            f"Les thèmes {', '.join(label(s) for s in vigilance[:2])} sont en zone de vigilance (40-60% négatif) et méritent un suivi renforcé."
        )
    alertes_texte = " ".join(alertes_parts) if alertes_parts else "Aucun thème critique détecté ce mois."

    # ── POSITIFS
    positifs_labels = [label(s) for s in positifs[:3]]
    if not positifs_labels:
        positifs_labels = ["Aucun thème majoritairement positif ce mois"]

    # ── ACTIONS RECOMMANDÉES
    actions = []
    if critiques:
        top = critiques[0]
        actions.append(
            f"Organiser un atelier de travail participatif sur le thème \"{label(top)}\" "
            f"avec les équipes concernées dans les 2 semaines."
        )
    if has_burnout:
        actions.append(
            "Déployer immédiatement une ligne d'écoute psychologique confidentielle "
            "et informer les collaborateurs de son existence."
        )
    if len(critiques) > 1:
        actions.append(
            f"Mettre en place un groupe de travail dédié à \"{label(critiques[1])}\" "
            "avec un plan d'action concret sous 30 jours."
        )
    if vigilance:
        actions.append(
            f"Lancer un sondage ciblé sur \"{label(vigilance[0])}\" pour identifier "
            "les causes racines avant que la situation ne se dégrade."
        )
    if positifs:
        actions.append(
            f"Capitaliser sur les bonnes pratiques de \"{label(positifs[0])}\" "
            "en les documentant et en les partageant à l'ensemble des équipes."
        )
    if not actions:
        actions.append("Maintenir le suivi mensuel et encourager la participation au sondage QVT.")
    actions = actions[:4]

    # ── TENDANCE
    if score >= 0.3:
        tendance = (
            f"Le climat social de {mois} est {tendance_mot}. "
            "Pour maintenir cette dynamique, continuez à valoriser les équipes et renforcez les pratiques positives identifiées."
        )
    elif score >= -0.3:
        tendance = (
            f"Le climat social de {mois} est {tendance_mot}. "
            f"Le mois prochain, concentrez les efforts sur {'les thèmes critiques identifiés' if critiques else 'la consolidation des acquis'} "
            "pour faire progresser le score global vers une zone positive."
        )
    else:
        tendance = (
            f"Le climat social de {mois} est {tendance_mot} et nécessite une intervention structurée. "
            "Une communication transparente de la direction sur les mesures prises est indispensable pour restaurer la confiance."
        )

    return {
        "resume":         resume,
        "alertes_themes": alertes_themes,
        "alertes_texte":  alertes_texte,
        "positifs":       positifs_labels,
        "actions":        actions,
        "tendance":       tendance,
    }


@app.route("/api/synthese", methods=["POST"])
def api_synthese():
    """
    Génère une synthèse narrative mensuelle RH.
    Utilise Claude API si ANTHROPIC_API_KEY est définie, sinon générateur local gratuit.
    Body JSON : { mois, score_global, alerte_burnout, stats_themes }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Body JSON requis"}), 400

    # ── Tentative via Claude API (si clé disponible et module installé)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if _ANTHROPIC_AVAILABLE and api_key:
        try:
            client  = _anthropic.Anthropic(api_key=api_key)
            prompt  = _build_synthese_prompt_llm(data)
            message = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 1024,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw_text = message.content[0].text.strip()
            if raw_text.startswith("```"):
                parts    = raw_text.split("```")
                raw_text = parts[1] if len(parts) >= 2 else raw_text
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            synthese = json.loads(raw_text.strip())
            synthese["source"] = "llm"
            return jsonify(synthese)
        except Exception as e:
            print(f"⚠️  Claude API indisponible ({e}) → bascule sur générateur local")

    # ── Générateur local (gratuit, toujours disponible)
    synthese = _generer_synthese_locale(data)
    synthese["source"] = "local"
    return jsonify(synthese)


def _build_synthese_prompt_llm(data: dict) -> str:
    """Prompt pour Claude API (utilisé uniquement si clé disponible)."""
    mois           = data.get("mois", "N/A")
    score_global   = data.get("score_global", {})
    alerte_burnout = data.get("alerte_burnout", {})
    stats_themes   = data.get("stats_themes", [])
    themes_lines   = []
    for s in stats_themes:
        total   = s.get("total", 0)
        pct_neg = s.get("pct_negatif", 0)
        pct_neu = round(s.get("nb_neutre", 0) / total * 100, 1) if total else 0
        pct_pos = round(s.get("nb_positif", 0) / total * 100, 1) if total else 0
        note    = f" | Note : {s.get('note_moyenne')}/5" if s.get("note_moyenne") else ""
        themes_lines.append(f"  - {s.get('label', s.get('theme'))} : {pct_neg}% négatif / {pct_neu}% neutre / {pct_pos}% positif ({total} retours){note}")
    burnout_info = (f"🚨 ALERTE : {alerte_burnout.get('nb_signaux', 0)} signaux de détresse."
                    if alerte_burnout.get("alerte")
                    else f"Signaux : {alerte_burnout.get('nb_signaux', 0)} (sous seuil).")
    themes_block = chr(10).join(themes_lines) if themes_lines else "  Aucune donnée suffisante."
    return f"""Expert QVT. Synthèse mensuelle pour {mois}.
Score sentiment : {score_global.get('score_global', 0):.2f} | Total : {score_global.get('total', 0)} retours | {burnout_info}
Thèmes :
{themes_block}
Réponds UNIQUEMENT en JSON sans markdown :
{{"resume":"...","alertes_themes":["..."],"alertes_texte":"...","positifs":["..."],"actions":["...","...","..."],"tendance":"..."}}"""

# =============================================================================
# ROUTES 7 : Gestion des modèles (admin)
# =============================================================================

@app.route("/api/model/set/<model_type>", methods=["POST"])
def set_model(model_type):
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
    return jsonify(get_model_info())


@app.route("/api/model/zeroshot/themes")
def get_zeroshot_themes():
    return jsonify({
        "themes":     THEMES_ZEROSHOT,
        "model_name": "BaptisteDoyen/camembert-base-xnli",
        "note":       "Modifiables à chaud via POST /api/model/zeroshot/themes/add"
    })


@app.route("/api/model/zeroshot/themes/add", methods=["POST"])
def add_zeroshot_theme():
    data  = request.get_json()
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