# 🧠 QVT Agent — Guide d'installation et d'utilisation

## Structure du projet

```
qvt_project/
│
├── app.py                    ← Serveur Flask (point d'entrée)
├── nlp_engine.py             ← Moteur NLP (thème + sentiment + burnout)
├── database.py               ← Base SQLite anonymisée
├── seed_data.py              ← Script pour peupler la BDD avec des données de test
├── requirements.txt          ← Dépendances Python
│
├── data/
│   ├── verbatims_entrainement.csv  ← Phrases d'entraînement (ton fichier Excel converti)
│   ├── lexique_sentiment.csv       ← Lexique de sentiment
│   ├── recommandations.json        ← Actions RH par thème
│   └── qvt.db                      ← Base SQLite (créée automatiquement)
│
├── models/
│   └── theme_classifier.pkl        ← Modèle entraîné (créé automatiquement)
│
└── templates/
    ├── formulaire.html     ← Page collaborateur
    ├── merci.html          ← Page de confirmation
    └── dashboard.html      ← Dashboard RH
```

---

## ⚙️ Installation (une seule fois)

### 1. Vérifie que Python 3.10+ est installé
```bash
python --version
# Doit afficher Python 3.10.x ou supérieur
```

### 2. Crée un environnement virtuel
```bash
# Dans le dossier qvt_project
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Mac/Linux)
source venv/bin/activate
```

### 3. Installe les dépendances
```bash
pip install -r requirements.txt
```

C'est tout. Pas de base de données externe, pas de serveur, tout tourne en local.

---

## 🚀 Lancement

### Étape 1 — Entraîner le modèle (première fois uniquement)
```bash
python nlp_engine.py
```
Tu verras les performances du classifieur s'afficher (precision, recall, F1).

### Étape 2 — Peupler la base avec des données de test (optionnel)
```bash
python seed_data.py
```
Insère ~50 retours réalistes pour voir le dashboard fonctionner immédiatement.

### Étape 3 — Lancer le serveur
```bash
python app.py
```

### Étape 4 — Ouvrir dans le navigateur
- **Formulaire collaborateur** → http://localhost:5000
- **Dashboard RH**             → http://localhost:5000/dashboard
- **API JSON**                  → http://localhost:5000/api/stats

---

## 🔄 Comment mettre à jour les données d'entraînement

Tes fichiers Excel ont les mêmes données, juste sous une forme différente.
Pour les utiliser, convertis-les en CSV avec ce format exact :

```
texte,theme,sentiment
"J'ai tellement de dossiers...",CHARGE,NEGATIF
"Mon manager ne donne jamais...",MGMT,NEGATIF
```

**Codes thèmes acceptés :** CHARGE, MGMT, OUTILS, FORMATION, EQUIPE, SALAIRE, REMOTE, WELLBEING
**Codes sentiment acceptés :** NEGATIF, NEUTRE, POSITIF

Remplace `data/verbatims_entrainement.csv` puis ré-entraîne :
```bash
python nlp_engine.py
# Le fichier models/theme_classifier.pkl sera mis à jour
```

---

## 📊 Comprendre le dashboard

### Score QVT global
- Entre -1.0 (très négatif) et +1.0 (très positif)
- Calculé comme la moyenne des scores de sentiment de tous les retours

### Barres par thème
- Rouge = % de retours négatifs sur ce thème
- Si une barre rouge dépasse 60% → recommandation automatique déclenchée

### Règle d'anonymat
- Un thème n'est affiché que si au moins 5 personnes ont répondu dessus
- La variable `N_MIN` dans `database.py` contrôle ce seuil

### Alerte burnout
- Déclenchée si 3+ retours contiennent des mots-clés de détresse ET un score < -0.55
- Toujours affiché de façon agrégée, jamais nominative

---

## 🔧 Personnalisation

| Ce que tu veux changer | Fichier à modifier |
|---|---|
| Ajouter des phrases d'entraînement | `data/verbatims_entrainement.csv` |
| Modifier les mots de sentiment | `data/lexique_sentiment.csv` |
| Changer les recommandations RH | `data/recommandations.json` |
| Changer le seuil d'anonymat (N=5) | `database.py` ligne `N_MIN = 5` |
| Changer le seuil de déclenchement reco (60%) | `data/recommandations.json` champ `"seuil"` |
| Changer le seuil burnout | `nlp_engine.py` fonction `detecter_burnout` |
| Modifier le design du formulaire | `templates/formulaire.html` |
| Modifier le design du dashboard | `templates/dashboard.html` |
