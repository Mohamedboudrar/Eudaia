"""
database.py
-----------
Gestion de la base de données SQLite.
Stocke UNIQUEMENT les données anonymisées (jamais le texte brut).
Applique la règle d'anonymat : N_MIN retours minimum pour afficher un thème.
"""

import sqlite3, os
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "qvt.db")

N_MIN = 5  # Nombre minimum de retours pour afficher un thème (anonymat)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Crée les tables si elles n'existent pas."""
    conn = get_connection()
    cur = conn.cursor()

    # Table principale des retours (anonymisée)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS retours (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date_envoi  TEXT    NOT NULL,
            semaine     TEXT    NOT NULL,   -- ex: "2025-W08"
            mois        TEXT    NOT NULL,   -- ex: "2025-02"
            theme       TEXT    NOT NULL,
            confiance   REAL    NOT NULL,
            sentiment   TEXT    NOT NULL,
            score       REAL    NOT NULL,
            signal_burnout INTEGER DEFAULT 0,
            note_quanti INTEGER             -- 1 à 5, peut être NULL
        )
    """)

    # Table compteur burnout (agrégat mensuel, jamais nominatif)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alertes_burnout (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            mois     TEXT    NOT NULL,
            nb_signaux INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

def sauvegarder_retour(analyse: dict, note_quanti: int = None):
    """
    Enregistre le résultat d'une analyse dans la base.
    Le texte brut n'est jamais passé ici.
    """
    now = datetime.now()
    semaine = now.strftime("%Y-W%W")
    mois    = now.strftime("%Y-%m")

    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        INSERT INTO retours
            (date_envoi, semaine, mois, theme, confiance, sentiment, score, signal_burnout, note_quanti)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now.isoformat(),
        semaine,
        mois,
        analyse["theme"],
        analyse["confiance"],
        analyse["sentiment"],
        analyse["score_sentiment"],
        1 if analyse["signal_burnout"] else 0,
        note_quanti
    ))

    # Mise à jour du compteur burnout si signal détecté
    if analyse["signal_burnout"]:
        cur.execute("SELECT id, nb_signaux FROM alertes_burnout WHERE mois = ?", (mois,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE alertes_burnout SET nb_signaux = nb_signaux + 1 WHERE mois = ?", (mois,))
        else:
            cur.execute("INSERT INTO alertes_burnout (mois, nb_signaux) VALUES (?, 1)", (mois,))

    conn.commit()
    conn.close()

# ── REQUÊTES DASHBOARD ────────────────────────────────────────────────────────

def get_stats_themes(mois: str = None) -> list[dict]:
    """
    Retourne les stats par thème pour un mois donné (ou tous les mois).
    Applique la règle d'anonymat : thèmes avec < N_MIN retours → masqués.
    """
    conn = get_connection()
    cur  = conn.cursor()

    if mois:
        cur.execute("""
            SELECT
                theme,
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'NEGATIF' THEN 1 ELSE 0 END) as nb_negatif,
                SUM(CASE WHEN sentiment = 'NEUTRE'  THEN 1 ELSE 0 END) as nb_neutre,
                SUM(CASE WHEN sentiment = 'POSITIF' THEN 1 ELSE 0 END) as nb_positif,
                AVG(score) as score_moyen,
                AVG(note_quanti) as note_moyenne
            FROM retours
            WHERE mois = ? AND theme != 'NON_CLASSE'
            GROUP BY theme
            HAVING COUNT(*) >= ?
            ORDER BY score_moyen ASC
        """, (mois, N_MIN))
    else:
        cur.execute("""
            SELECT
                theme,
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'NEGATIF' THEN 1 ELSE 0 END) as nb_negatif,
                SUM(CASE WHEN sentiment = 'NEUTRE'  THEN 1 ELSE 0 END) as nb_neutre,
                SUM(CASE WHEN sentiment = 'POSITIF' THEN 1 ELSE 0 END) as nb_positif,
                AVG(score) as score_moyen,
                AVG(note_quanti) as note_moyenne
            FROM retours
            WHERE theme != 'NON_CLASSE'
            GROUP BY theme
            HAVING COUNT(*) >= ?
            ORDER BY score_moyen ASC
        """, (N_MIN,))

    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        total = r["total"]
        pct_neg = round(r["nb_negatif"] / total * 100, 1) if total > 0 else 0
        result.append({
            "theme":        r["theme"],
            "total":        total,
            "nb_negatif":   r["nb_negatif"],
            "nb_neutre":    r["nb_neutre"],
            "nb_positif":   r["nb_positif"],
            "pct_negatif":  pct_neg,
            "score_moyen":  round(r["score_moyen"], 3) if r["score_moyen"] else 0,
            "note_moyenne": round(r["note_moyenne"], 1) if r["note_moyenne"] else None,
        })
    return result

def get_tendance(nb_mois: int = 3) -> list[dict]:
    """
    Retourne l'évolution du score moyen par thème sur les nb_mois derniers mois.
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT mois, theme, AVG(score) as score_moyen, COUNT(*) as total
        FROM retours
        WHERE theme != 'NON_CLASSE'
        GROUP BY mois, theme
        HAVING COUNT(*) >= ?
        ORDER BY mois ASC
    """, (N_MIN,))
    rows = cur.fetchall()
    conn.close()

    # Réorganiser par thème → liste de mois
    tendances = {}
    for r in rows:
        t = r["theme"]
        if t not in tendances:
            tendances[t] = []
        tendances[t].append({
            "mois":  r["mois"],
            "score": round(r["score_moyen"], 3),
            "total": r["total"]
        })

    return tendances

def get_score_global(mois: str = None) -> dict:
    """Score QVT global = moyenne de tous les scores du mois."""
    conn = get_connection()
    cur  = conn.cursor()

    if mois:
        cur.execute("""
            SELECT AVG(score) as score_global, COUNT(*) as total,
                   AVG(note_quanti) as note_globale
            FROM retours WHERE mois = ? AND theme != 'NON_CLASSE'
        """, (mois,))
    else:
        cur.execute("""
            SELECT AVG(score) as score_global, COUNT(*) as total,
                   AVG(note_quanti) as note_globale
            FROM retours WHERE theme != 'NON_CLASSE'
        """)

    r = cur.fetchone()
    conn.close()
    return {
        "score_global": round(r["score_global"], 3) if r["score_global"] else 0,
        "total":        r["total"] or 0,
        "note_globale": round(r["note_globale"], 1) if r["note_globale"] else None,
    }

def get_alerte_burnout(mois: str = None) -> dict:
    """Retourne le nb de signaux burnout pour le mois. Toujours anonyme."""
    conn = get_connection()
    cur  = conn.cursor()

    if mois:
        cur.execute("SELECT nb_signaux FROM alertes_burnout WHERE mois = ?", (mois,))
    else:
        cur.execute("SELECT SUM(nb_signaux) as nb_signaux FROM alertes_burnout")

    r = cur.fetchone()
    conn.close()
    nb = r["nb_signaux"] if r and r["nb_signaux"] else 0
    return {"nb_signaux": nb, "alerte": nb >= 3}

def get_mois_disponibles() -> list[str]:
    """Liste des mois qui ont des données."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT DISTINCT mois FROM retours ORDER BY mois DESC")
    rows = cur.fetchall()
    conn.close()
    return [r["mois"] for r in rows]

# Initialisation automatique au premier import
init_db()
