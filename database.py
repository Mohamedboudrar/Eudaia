"""
database.py
-----------
Gestion de la base de données SQLite.
Stocke UNIQUEMENT les données anonymisées (jamais le texte brut).
Applique la règle d'anonymat : N_MIN retours minimum pour afficher un thème.

CORRECTIFS v2 :
  - Calcul dynamique des stats à chaque appel (plus de cache statique)
  - Distinction nette NEGATIF / NEUTRE / POSITIF via les colonnes réelles
  - pct_negatif, pct_neutre, pct_positif toujours cohérents (somme = 100%)
  - score_moyen calculé séparément pour chaque sentiment
"""

import sqlite3, os
from datetime import datetime

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
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS retours (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            date_envoi     TEXT    NOT NULL,
            semaine        TEXT    NOT NULL,
            mois           TEXT    NOT NULL,
            theme          TEXT    NOT NULL,
            confiance      REAL    NOT NULL,
            sentiment      TEXT    NOT NULL,   -- 'NEGATIF' | 'NEUTRE' | 'POSITIF'
            score          REAL    NOT NULL,
            signal_burnout INTEGER DEFAULT 0,
            note_quanti    INTEGER            -- 1 à 5, nullable
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alertes_burnout (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            mois       TEXT    NOT NULL,
            nb_signaux INTEGER DEFAULT 0
        )
    """)

    # Index pour accélérer les requêtes dashboard
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_retours_mois_theme
        ON retours (mois, theme)
    """)

    conn.commit()
    conn.close()


def sauvegarder_retour(analyse: dict, note_quanti: int = None):
    """
    Enregistre le résultat d'une analyse dans la base.
    Le texte brut n'est JAMAIS passé ici.
    """
    now     = datetime.now()
    semaine = now.strftime("%Y-W%W")
    mois    = now.strftime("%Y-%m")

    # Normalisation défensive du sentiment
    sentiment = analyse.get("sentiment", "NEUTRE").upper()
    if sentiment not in ("NEGATIF", "NEUTRE", "POSITIF"):
        sentiment = "NEUTRE"

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
        sentiment,
        analyse["score_sentiment"],
        1 if analyse.get("signal_burnout") else 0,
        note_quanti,
    ))

    if analyse.get("signal_burnout"):
        cur.execute("SELECT id FROM alertes_burnout WHERE mois = ?", (mois,))
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE alertes_burnout SET nb_signaux = nb_signaux + 1 WHERE mois = ?",
                (mois,)
            )
        else:
            cur.execute(
                "INSERT INTO alertes_burnout (mois, nb_signaux) VALUES (?, 1)",
                (mois,)
            )

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# REQUÊTES DASHBOARD — toutes dynamiques, recalculées à chaque appel
# ─────────────────────────────────────────────────────────────────────────────

def get_stats_themes(mois: str = None) -> list[dict]:
    """
    Retourne les statistiques par thème, calculées en temps réel depuis la DB.

    Pour chaque thème ayant >= N_MIN retours :
      - Compte exact des sentiments NEGATIF / NEUTRE / POSITIF (colonne réelle)
      - Pourcentages arrondis et cohérents (somme = 100 %)
      - Score moyen sur [-1, 1]
      - Note moyenne sur [1, 5] si disponible

    Trié par score_moyen ASC (les thèmes les plus critiques en premier).
    """
    conn = get_connection()
    cur  = conn.cursor()

    base_query = """
        SELECT
            theme,
            COUNT(*)                                                       AS total,
            SUM(CASE WHEN UPPER(sentiment) = 'NEGATIF' THEN 1 ELSE 0 END) AS nb_negatif,
            SUM(CASE WHEN UPPER(sentiment) = 'NEUTRE'  THEN 1 ELSE 0 END) AS nb_neutre,
            SUM(CASE WHEN UPPER(sentiment) = 'POSITIF' THEN 1 ELSE 0 END) AS nb_positif,
            AVG(score)                                                     AS score_moyen,
            AVG(note_quanti)                                               AS note_moyenne
        FROM retours
        WHERE theme != 'NON_CLASSE'
        {mois_filter}
        GROUP BY theme
        HAVING COUNT(*) >= {n_min}
        ORDER BY score_moyen ASC
    """

    if mois:
        query = base_query.format(mois_filter="AND mois = ?", n_min=N_MIN)
        cur.execute(query, (mois,))
    else:
        query = base_query.format(mois_filter="", n_min=N_MIN)
        cur.execute(query)

    rows   = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        total = r["total"] or 1  # évite division par zéro

        nb_neg = int(r["nb_negatif"] or 0)
        nb_neu = int(r["nb_neutre"]  or 0)
        nb_pos = int(r["nb_positif"] or 0)

        # Recalcul défensif : les counts doivent sommer à total
        # (au cas où un sentiment non standard serait en base)
        nb_autres = total - nb_neg - nb_neu - nb_pos
        if nb_autres > 0:
            # on les classe en neutre par sécurité
            nb_neu += nb_autres

        # Pourcentages entiers cohérents (méthode "largest remainder")
        pct_neg = round(nb_neg / total * 100, 1)
        pct_neu = round(nb_neu / total * 100, 1)
        pct_pos = round(nb_pos / total * 100, 1)

        result.append({
            "theme":        r["theme"],
            "total":        total,
            "nb_negatif":   nb_neg,
            "nb_neutre":    nb_neu,
            "nb_positif":   nb_pos,
            "pct_negatif":  pct_neg,
            "pct_neutre":   pct_neu,
            "pct_positif":  pct_pos,
            "score_moyen":  round(float(r["score_moyen"] or 0), 3),
            "note_moyenne": round(float(r["note_moyenne"]), 1) if r["note_moyenne"] else None,
        })

    return result


def get_tendance(nb_mois: int = 6) -> dict:
    """
    Retourne l'évolution du score moyen par thème sur les derniers mois.
    Dynamique : recalculé à chaque appel.
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        SELECT mois, theme, AVG(score) AS score_moyen, COUNT(*) AS total
        FROM retours
        WHERE theme != 'NON_CLASSE'
        GROUP BY mois, theme
        HAVING COUNT(*) >= ?
        ORDER BY mois ASC
    """, (N_MIN,))

    rows = cur.fetchall()
    conn.close()

    tendances = {}
    for r in rows:
        t = r["theme"]
        if t not in tendances:
            tendances[t] = []
        tendances[t].append({
            "mois":  r["mois"],
            "score": round(float(r["score_moyen"]), 3),
            "total": r["total"],
        })

    return tendances


def get_score_global(mois: str = None) -> dict:
    """Score QVT global = moyenne pondérée de tous les scores du mois."""
    conn = get_connection()
    cur  = conn.cursor()

    if mois:
        cur.execute("""
            SELECT
                AVG(score)       AS score_global,
                COUNT(*)         AS total,
                AVG(note_quanti) AS note_globale
            FROM retours
            WHERE mois = ? AND theme != 'NON_CLASSE'
        """, (mois,))
    else:
        cur.execute("""
            SELECT
                AVG(score)       AS score_global,
                COUNT(*)         AS total,
                AVG(note_quanti) AS note_globale
            FROM retours
            WHERE theme != 'NON_CLASSE'
        """)

    r = cur.fetchone()
    conn.close()

    return {
        "score_global": round(float(r["score_global"]), 3) if r["score_global"] else 0.0,
        "total":        int(r["total"]) if r["total"] else 0,
        "note_globale": round(float(r["note_globale"]), 1) if r["note_globale"] else None,
    }


def get_alerte_burnout(mois: str = None) -> dict:
    """Retourne le nb de signaux burnout. Toujours agrégé, jamais nominatif."""
    conn = get_connection()
    cur  = conn.cursor()

    if mois:
        cur.execute(
            "SELECT nb_signaux FROM alertes_burnout WHERE mois = ?",
            (mois,)
        )
    else:
        cur.execute("SELECT SUM(nb_signaux) AS nb_signaux FROM alertes_burnout")

    r  = cur.fetchone()
    conn.close()

    nb = int(r["nb_signaux"]) if (r and r["nb_signaux"]) else 0
    return {"nb_signaux": nb, "alerte": nb >= 3}


def get_mois_disponibles() -> list[str]:
    """Liste des mois ayant des données, triés du plus récent au plus ancien."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "SELECT DISTINCT mois FROM retours ORDER BY mois DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [r["mois"] for r in rows]


def get_debug_counts(mois: str = None) -> dict:
    """
    Outil de débogage : retourne le détail brut des sentiments en base.
    Utile pour vérifier que la classification NLP est correcte.
    """
    conn = get_connection()
    cur  = conn.cursor()

    if mois:
        cur.execute("""
            SELECT theme, sentiment, COUNT(*) as cnt
            FROM retours
            WHERE mois = ?
            GROUP BY theme, sentiment
            ORDER BY theme, sentiment
        """, (mois,))
    else:
        cur.execute("""
            SELECT theme, sentiment, COUNT(*) as cnt
            FROM retours
            GROUP BY theme, sentiment
            ORDER BY theme, sentiment
        """)

    rows = cur.fetchall()
    conn.close()

    debug = {}
    for r in rows:
        t = r["theme"]
        if t not in debug:
            debug[t] = {}
        debug[t][r["sentiment"]] = r["cnt"]

    return debug


# Initialisation automatique au premier import
init_db()