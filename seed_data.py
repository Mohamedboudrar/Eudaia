"""
seed_data.py
------------
Peuple la base avec des données de test réalistes
pour pouvoir visualiser le dashboard immédiatement.

Lance avec : python seed_data.py [--model logistic|naive_bayes|zeroshot]

Exemples :
  python seed_data.py                        # utilise logistic (défaut)
  python seed_data.py --model naive_bayes    # utilise Naive Bayes
  python seed_data.py --model zeroshot       # utilise CamemBERT-XNLI zero-shot
"""

import sys, os, argparse, random, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_engine import (analyser, entrainer_modele, set_active_model,
                        get_active_model, MODELS_DISPONIBLES)
from database import sauvegarder_retour, init_db
from datetime import datetime, timedelta

# ── Phrases de test représentatives (couvrant les 8 thèmes RH) ───────────────
PHRASES_TEST = [
    # CHARGE DE TRAVAIL — négatif
    ("Je suis complètement débordé depuis 3 semaines, je travaille jusqu'à 21h tous les soirs.", 1),
    ("Les deadlines s'accumulent et on n'a pas les ressources pour tenir.", 2),
    ("On est en sous-effectif et personne ne semble s'en préoccuper.", 1),
    ("Ma to-do list est impossible à finir, je stagne tout le temps.", 2),
    ("Trop de réunions inutiles, je n'ai plus le temps de travailler vraiment.", 2),
    ("Je suis épuisé, je n'arrive plus à récupérer le weekend.", 1),
    ("La charge de travail est totalement ingérable depuis le dernier projet.", 2),
    # CHARGE DE TRAVAIL — positif
    ("La charge est bien répartie dans l'équipe, je me sens stimulé sans être débordé.", 5),
    ("Mon manager sait dire non, on n'est jamais surchargés inutilement.", 4),

    # MANAGEMENT — négatif
    ("Mon manager ne nous donne jamais de retour sur notre travail.", 2),
    ("Il y a clairement du favoritisme dans les décisions de promotion.", 1),
    ("Les objectifs changent toutes les semaines, c'est impossible à suivre.", 2),
    ("Mon chef contrôle tout, je n'ai aucune autonomie.", 1),
    ("Je n'ai pas eu d'entretien individuel depuis plus d'un an.", 2),
    # MANAGEMENT — positif
    ("Mon manager m'écoute vraiment et me fait confiance, c'est rare.", 5),
    ("J'ai un one-to-one hebdomadaire très utile avec mon responsable.", 4),

    # OUTILS — négatif
    ("L'ERP plante plusieurs fois par jour, je perds un temps fou.", 2),
    ("Le VPN coupe toutes les 30 minutes en télétravail, c'est ingérable.", 1),
    ("On travaille avec des logiciels obsolètes depuis 10 ans.", 2),
    ("Les accès ne fonctionnent pas, j'ai dû redemander 3 fois cette semaine.", 1),
    # OUTILS — positif
    ("Les outils sont modernes et vraiment adaptés, ça facilite le travail.", 5),

    # FORMATION/ÉVOLUTION — négatif
    ("J'ai demandé une formation il y a 6 mois et toujours pas de réponse.", 2),
    ("Je fais le même poste depuis 4 ans, je stagne complètement.", 1),
    ("Le budget formation a été supprimé sans explication.", 1),
    ("Aucune perspective d'évolution malgré mes demandes répétées.", 2),
    # FORMATION/ÉVOLUTION — positif
    ("L'entreprise m'a permis de suivre une certification, c'est top.", 5),
    ("Mon manager m'encourage activement à me former.", 4),

    # ÉQUIPE/AMBIANCE — négatif
    ("Il y a beaucoup de tensions dans l'équipe en ce moment.", 2),
    ("Je me sens mis à l'écart du groupe depuis quelque temps.", 1),
    ("L'ambiance est vraiment mauvaise à cause de deux collègues en conflit.", 1),
    ("On ne s'entraide plus du tout, chacun pour soi.", 2),
    # ÉQUIPE/AMBIANCE — positif
    ("Super ambiance dans l'équipe, on s'entraide vraiment bien.", 5),
    ("Mes collègues sont toujours là quand j'ai besoin d'aide.", 4),
    ("On a fait un team building génial la semaine dernière.", 5),

    # SALAIRE/RÉMUNÉRATION — négatif
    ("Je suis payé en dessous du marché depuis 3 ans.", 1),
    ("On m'a promis une augmentation et elle n'est jamais arrivée.", 2),
    ("Les critères d'augmentation ne sont jamais expliqués, tout est opaque.", 2),
    ("Avec l'inflation mon salaire réel baisse chaque année.", 1),
    # SALAIRE/RÉMUNÉRATION — positif
    ("J'ai eu une augmentation surprise cette année, très appréciée.", 5),
    ("Le package global est vraiment compétitif.", 4),

    # TÉLÉTRAVAIL/ÉQUILIBRE — négatif
    ("On nous a retiré le télétravail sans aucune justification.", 1),
    ("Mon manager envoie des messages à 22h et attend une réponse.", 1),
    ("Je passe 3 heures dans les transports alors que je pourrais télétravailler.", 2),
    ("La politique télétravail est floue et dépend de chaque manager.", 2),
    # TÉLÉTRAVAIL/ÉQUILIBRE — positif
    ("La flexibilité horaire change vraiment ma qualité de vie.", 5),
    ("On peut déconnecter sereinement après 18h, personne ne t'appelle.", 5),
    ("La politique hybride est claire et vraiment respectée.", 4),

    # BIEN-ÊTRE — négatif
    ("Je n'arrive plus à dormir tellement je pense au travail le soir.", 1),
    ("Plusieurs collègues ont craqué cette année et rien n'est fait.", 1),
    ("Je n'ai plus envie de venir travailler le matin.", 1),
    ("La pression est insupportable, je suis à bout.", 1),
    # BIEN-ÊTRE — burnout signal
    ("Je suis à bout, je ne peux plus continuer comme ça. Je pense à démissionner.", 1),
    ("J'ai l'impression de craquer, je n'en peux vraiment plus.", 1),
    # BIEN-ÊTRE — positif
    ("Je ressors du travail avec de l'énergie, c'est bon signe.", 4),
    ("L'ambiance bienveillante contribue vraiment à mon bien-être.", 5),
]


def seed(model_type: str = "logistic"):
    """Insère les données de test dans la base en utilisant le moteur spécifié."""
    print(f"🌱 Seeding de la base de données (moteur : {model_type})...")
    init_db()

    # Prépare le moteur NLP
    if model_type in ["logistic", "naive_bayes"]:
        entrainer_modele(model_type=model_type)
    set_active_model(model_type)

    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "data", "qvt.db"))
    cur  = conn.cursor()

    now            = datetime.now()
    mois_courant   = now.strftime("%Y-%m")
    mois_precedent = (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")

    inserted = 0
    skipped  = 0

    for phrase, note in PHRASES_TEST:
        try:
            resultat = analyser(phrase)
        except Exception as e:
            print(f"  ⚠️  Erreur NLP : {e} — phrase ignorée")
            skipped += 1
            continue

        if resultat["theme"] == "NON_CLASSE":
            skipped += 1
            continue

        mois       = mois_courant if random.random() > 0.4 else mois_precedent
        date_fake  = f"{mois}-{random.randint(1, 28):02d}T{random.randint(8, 18):02d}:00:00"
        semaine    = f"{mois}-W{random.randint(1, 4):02d}"

        cur.execute("""
            INSERT INTO retours
                (date_envoi, semaine, mois, theme, confiance, sentiment,
                 score, signal_burnout, note_quanti)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_fake, semaine, mois,
            resultat["theme"], resultat["confiance"],
            resultat["sentiment"], resultat["score_sentiment"],
            1 if resultat["signal_burnout"] else 0,
            note,
        ))

        # Gestion des alertes burnout
        if resultat["signal_burnout"]:
            cur.execute("SELECT id FROM alertes_burnout WHERE mois = ?", (mois,))
            if cur.fetchone():
                cur.execute(
                    "UPDATE alertes_burnout SET nb_signaux = nb_signaux + 1 WHERE mois = ?", (mois,)
                )
            else:
                cur.execute(
                    "INSERT INTO alertes_burnout (mois, nb_signaux) VALUES (?, 1)", (mois,)
                )

        inserted += 1
        burnout_flag = " 🚨" if resultat["signal_burnout"] else ""
        print(f"  [{resultat['theme']:20}] [{resultat['sentiment']:8}] "
              f"conf={resultat['confiance']:.2f}{burnout_flag}  {phrase[:55]}...")

    conn.commit()
    conn.close()

    print(f"\n✅ {inserted} retours insérés ({skipped} ignorés).")
    print(f"   Moteur utilisé : {model_type}")
    print("   Lance maintenant : python app.py")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed QVT database")
    parser.add_argument(
        "--model",
        choices=MODELS_DISPONIBLES,
        default="logistic",
        help=f"Moteur NLP à utiliser : {MODELS_DISPONIBLES} (défaut: logistic)"
    )
    args = parser.parse_args()
    seed(model_type=args.model)