"""
app.py — QVT Agent v2
---------------------
CORRECTIFS :
  - Après /soumettre : redirect vers /merci avec timestamp pour éviter le cache
  - /dashboard : stats toujours recalculées depuis la DB (pas de cache serveur)
  - /api/debug  : endpoint de vérification des données brutes en base
  - Exposition des pct_neutre / pct_positif calculés dans database.py
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from datetime import datetime
import json, os

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass

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
    get_debug_counts,
)

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

app = Flask(__name__)
DEFAULT_MODEL = "logistic"

# =============================================================================
# INITIALISATION
# =============================================================================

print("🚀 Démarrage QVT Agent v2...")

entrainer_modele(model_type="all")
print("✅ Modèles TF-IDF prêts.")

from nlp_engine import _ACTIVE_MODEL_FILE as _AMF
if not os.path.exists(_AMF):
    set_active_model(DEFAULT_MODEL)
    print(f"   Moteur par défaut : {DEFAULT_MODEL}")
else:
    print(f"   Moteur persisté : {get_active_model()}")

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
# Toujours rediriger vers /merci — ne JAMAIS afficher le résultat de l'analyse
# =============================================================================

@app.route("/soumettre", methods=["POST"])
def soumettre():
    texte    = request.form.get("verbatim", "").strip()
    note_str = request.form.get("note", "")

    if len(texte) < 10:
        return redirect(url_for("formulaire"))

    note_quanti = int(note_str) if note_str.isdigit() and 1 <= int(note_str) <= 5 else None

    # Analyse NLP (thème + sentiment + burnout)
    resultat = analyser(texte, note_quanti)

    # Sauvegarde anonymisée (texte brut jamais stocké)
    sauvegarder_retour(resultat, note_quanti)

    return redirect(url_for("merci"))

# =============================================================================
# ROUTE 3 : Confirmation
# =============================================================================

@app.route("/merci")
def merci():
    return render_template("merci.html")

# =============================================================================
# ROUTE 4 : Dashboard RH
# Les stats sont TOUJOURS relues depuis la DB à chaque requête GET.
# Aucun cache côté serveur — garantit des données fraîches après chaque soumission.
# =============================================================================

@app.route("/dashboard")
def dashboard():
    # Lecture fraîche des mois disponibles
    mois_dispo = get_mois_disponibles()
    mois_selec = request.args.get("mois", mois_dispo[0] if mois_dispo else None)

    # ── Stats dynamiques — relues depuis SQLite à chaque visite ──────────────
    stats_themes   = get_stats_themes(mois_selec)      # recalcul temps réel
    score_global   = get_score_global(mois_selec)       # idem
    alerte_burnout = get_alerte_burnout(mois_selec)     # idem
    tendances      = get_tendance()                      # idem

    # ── Enrichissement avec labels/couleurs depuis recommandations.json ───────
    for stat in stats_themes:
        info = RECOS.get(stat["theme"], {})
        stat["label"] = info.get("label", stat["theme"].replace("_", " ").title())
        stat["emoji"] = info.get("emoji", "📊")
        stat["color"] = info.get("color", "#888888")

        # Niveau d'urgence basé sur le % négatif réel (calculé en DB)
        pct = stat["pct_negatif"]
        if pct >= 70:
            stat["urgence"]       = "critique"
            stat["urgence_label"] = "🔴 CRITIQUE"
        elif pct >= 60:
            stat["urgence"]       = "haute"
            stat["urgence_label"] = "🔴 HAUTE"
        elif pct >= 40:
            stat["urgence"]       = "vigilance"
            stat["urgence_label"] = "🟡 VIGILANCE"
        else:
            stat["urgence"]       = "ok"
            stat["urgence_label"] = "🟢 OK"

    # ── Recommandations (seulement si pct_negatif >= seuil JSON) ─────────────
    recommandations_actives = []
    for stat in stats_themes:
        pct  = stat["pct_negatif"] / 100
        reco = get_recommandation(stat["theme"], pct)
        if reco:
            recommandations_actives.append({
                "theme":          stat["theme"],
                "label":          reco.get("label", stat["label"]),
                "emoji":          reco.get("emoji", stat["emoji"]),
                "color":          reco.get("color", stat["color"]),
                "pct_negatif":    stat["pct_negatif"],
                "action_directe": reco.get("action_directe", ""),
                "actions":        reco.get("actions", {}),
                "kpi":            reco.get("kpi", []),
                "risque":         reco.get("risque", ""),
            })

    # ── Données graphique tendance ────────────────────────────────────────────
    chart_tendances = {}
    for theme, points in tendances.items():
        info = RECOS.get(theme, {})
        chart_tendances[theme] = {
            "label":  info.get("label", theme.replace("_", " ").title()),
            "color":  info.get("color", "#888888"),
            "points": points,
        }

    model_info = get_model_info()

    response = make_response(render_template(
        "dashboard.html",
        stats_themes           = stats_themes,
        score_global           = score_global,
        alerte_burnout         = alerte_burnout,
        recommandations        = recommandations_actives,
        chart_tendances        = json.dumps(chart_tendances),
        mois_dispo             = mois_dispo,
        mois_selec             = mois_selec,
        model_info             = model_info,
        now_str                = datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    ))
    # CORRECTIF : empêche le navigateur de mettre en cache le dashboard
    # pour que les nouvelles soumissions soient immédiatement visibles
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response

# =============================================================================
# ROUTE 5 : API JSON stats (toujours fraîche)
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
# ROUTE 6 : Debug — vérification des données brutes (admin only)
# =============================================================================

@app.route("/api/debug")
def api_debug():
    """
    Retourne le détail brut des sentiments enregistrés en base.
    Permet de vérifier que la classification NLP est correcte.
    Exemple : GET /api/debug?mois=2025-02
    """
    mois = request.args.get("mois")
    return jsonify({
        "detail_sentiments": get_debug_counts(mois),
        "stats_dashboard":   get_stats_themes(mois),
        "score_global":      get_score_global(mois),
    })

# =============================================================================
# ROUTE 7 : Synthèse mensuelle LLM / locale
# =============================================================================

def _generer_synthese_locale(data: dict) -> dict:
    mois           = data.get("mois", "N/A")
    score_global   = data.get("score_global", {})
    alerte_burnout = data.get("alerte_burnout", {})
    stats_themes   = data.get("stats_themes", [])

    score      = score_global.get("score_global", 0)
    total      = score_global.get("total", 0)
    nb_signaux = alerte_burnout.get("nb_signaux", 0)
    has_burnout= alerte_burnout.get("alerte", False)

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

    critiques = [s for s in stats_themes if s.get("pct_negatif", 0) >= 60]
    vigilance = [s for s in stats_themes if 40 <= s.get("pct_negatif", 0) < 60]
    positifs  = [s for s in stats_themes if s.get("pct_negatif", 0) < 30
                 and s.get("nb_positif", 0) / max(s.get("total", 1), 1) >= 0.5]

    critiques.sort(key=lambda x: x.get("pct_negatif", 0), reverse=True)
    positifs.sort(key=lambda x: x.get("nb_positif", 0), reverse=True)

    def label(s):
        return s.get("label", s.get("theme", ""))

    resume_parts = [
        f"Le mois de {mois} enregistre {total} retours collaborateurs avec un sentiment "
        f"{qualif_score} (score : {score:+.2f}/1.0)."
    ]
    if critiques:
        noms = ", ".join(label(s) for s in critiques[:2])
        resume_parts.append(
            f"{len(critiques)} thème(s) atteignent un taux de négativité critique : {noms}."
        )
    if has_burnout:
        resume_parts.append(
            f"⚠️ {nb_signaux} signal(aux) de détresse détectés — attention immédiate requise."
        )
    if positifs:
        resume_parts.append(
            f"Points positifs : {', '.join(label(s) for s in positifs[:2])} affichent "
            f"des retours majoritairement satisfaisants."
        )

    alertes_themes = [label(s) for s in critiques[:3]]
    alertes_parts  = []
    if critiques:
        top = critiques[0]
        alertes_parts.append(
            f"Le thème \"{label(top)}\" concentre {top.get('pct_negatif', 0)}% de retours négatifs "
            f"sur {top.get('total', 0)} réponses."
        )
    if has_burnout:
        alertes_parts.append(
            f"{nb_signaux} signal(aux) de détresse nécessite un dispositif d'écoute confidentiel immédiat."
        )

    positifs_labels = [label(s) for s in positifs[:3]] or ["Aucun thème majoritairement positif ce mois"]

    actions = []
    if critiques:
        actions.append(
            f"Organiser un atelier participatif sur \"{label(critiques[0])}\" dans les 2 semaines."
        )
    if has_burnout:
        actions.append("Déployer une ligne d'écoute psychologique confidentielle immédiatement.")
    if len(critiques) > 1:
        actions.append(
            f"Plan d'action sur \"{label(critiques[1])}\" sous 30 jours."
        )
    if vigilance:
        actions.append(
            f"Sondage ciblé sur \"{label(vigilance[0])}\" pour identifier les causes racines."
        )
    if positifs:
        actions.append(
            f"Capitaliser sur les bonnes pratiques de \"{label(positifs[0])}\"."
        )
    if not actions:
        actions.append("Maintenir le suivi mensuel et encourager la participation au sondage QVT.")

    tendance_txt = (
        f"Le climat social de {mois} est {tendance_mot}. "
        + (
            "Continuez à valoriser les équipes et renforcez les pratiques positives."
            if score >= 0.3 else
            f"Concentrez les efforts sur {'les thèmes critiques identifiés' if critiques else 'la consolidation des acquis'}."
        )
    )

    return {
        "resume":         " ".join(resume_parts),
        "alertes_themes": alertes_themes,
        "alertes_texte":  " ".join(alertes_parts) or "Aucun thème critique détecté ce mois.",
        "positifs":       positifs_labels,
        "actions":        actions[:4],
        "tendance":       tendance_txt,
    }


@app.route("/api/synthese", methods=["POST"])
def api_synthese():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Body JSON requis"}), 400

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
            print(f"⚠️  Claude API indisponible ({e}) → bascule locale")

    synthese = _generer_synthese_locale(data)
    synthese["source"] = "local"
    return jsonify(synthese)


def _build_synthese_prompt_llm(data: dict) -> str:
    mois           = data.get("mois", "N/A")
    score_global   = data.get("score_global", {})
    alerte_burnout = data.get("alerte_burnout", {})
    stats_themes   = data.get("stats_themes", [])

    themes_lines = []
    for s in stats_themes:
        total   = s.get("total", 0)
        pct_neg = s.get("pct_negatif", 0)
        pct_neu = s.get("pct_neutre",  0)
        pct_pos = s.get("pct_positif", 0)
        note    = f" | Note : {s.get('note_moyenne')}/5" if s.get("note_moyenne") else ""
        themes_lines.append(
            f"  - {s.get('label', s.get('theme'))}: "
            f"{pct_neg}% négatif / {pct_neu}% neutre / {pct_pos}% positif "
            f"({total} retours){note}"
        )

    burnout_info = (
        f"🚨 ALERTE : {alerte_burnout.get('nb_signaux', 0)} signaux de détresse."
        if alerte_burnout.get("alerte")
        else f"Signaux : {alerte_burnout.get('nb_signaux', 0)} (sous seuil)."
    )

    return f"""Expert QVT. Synthèse mensuelle pour {mois}.
Score sentiment : {score_global.get('score_global', 0):.2f} | Total : {score_global.get('total', 0)} retours | {burnout_info}
Thèmes :
{chr(10).join(themes_lines) if themes_lines else "  Aucune donnée suffisante."}
Réponds UNIQUEMENT en JSON sans markdown :
{{"resume":"...","alertes_themes":["..."],"alertes_texte":"...","positifs":["..."],"actions":["...","...","..."],"tendance":"..."}}"""


# =============================================================================
# ROUTES 8 : Gestion des modèles
# =============================================================================

@app.route("/api/model/set/<model_type>", methods=["POST"])
def set_model(model_type):
    if model_type not in MODELS_DISPONIBLES:
        return jsonify({"error": f"Valeurs acceptées : {MODELS_DISPONIBLES}"}), 400
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
        "themes":   THEMES_ZEROSHOT,
        "model":    "joeddav/xlm-roberta-large-xnli",
    })


@app.route("/api/model/zeroshot/themes/add", methods=["POST"])
def add_zeroshot_theme():
    data  = request.get_json()
    cle   = data.get("cle", "").strip().upper()
    label = data.get("label", "").strip()
    if not cle or not label:
        return jsonify({"error": "Champs 'cle' et 'label' requis."}), 400
    ajouter_theme_zeroshot(cle, label)
    return jsonify({"status": "success", "theme": {cle: label}})


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5008)