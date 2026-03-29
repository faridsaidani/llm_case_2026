# Analyse CSAT - Cas Technique (Entretien)

## Objectif
Ce projet présente une analyse complète de feedbacks clients CSAT/NPS dans le cadre d'un entretien technique.

L'approche suit une logique pragmatique:
1. Extraire rapidement des insights avec du NLP classique (coût faible, interprétable)
2. Poser une baseline ML supervisée
3. Ajouter une analyse thématique non supervisée (LDA)
4. Enrichir un sous-ensemble avec un LLM (optionnel) pour une lecture métier plus fine

## Ce qui a été fait
Le script principal `analyse_csat_complete_standalone.py` exécute un pipeline en 6 étapes:
1. Chargement des données depuis SQLite (`db.sqlite`, table `csat_extract`)
2. Contrôles qualité et exploration des distributions CSAT
3. Analyse NLP classique sur verbatims négatifs (termes, bigrams, trigrams, TF-IDF)
4. Baseline de classification négatif/positif (TF-IDF + Logistic Regression)
5. Topic modeling (LDA) sur les verbatims négatifs
6. Enrichissement LLM (GPT) optionnel avec taxonomie métier fixe

Un dashboard Streamlit (`dashboard_llm_streamlit.py`) permet ensuite d'explorer les résultats LLM de manière interactive.

## Résultats clés
Source: `output_script/rapport_synthese.txt`

- Volume de données: 38 303 lignes
- Qualité:
  - commentaires vides: 1
  - NPS invalides: 3 677
  - CSAT invalides: 0
- Baseline classification (négatif vs positif): (**rajoutée pour qvoir un peu plus d'insights mais pas necessaire**) 
  - Accuracy: 0.7977
  - Precision: 0.4536
  - Recall: 0.8206
  - F1: 0.5842
  - AUC: 0.8811
- LDA:
  - 8 topics identifiés
  - perplexité: 1155.43
- Enrichissement LLM (échantillon):
  - 100 verbatims enrichis
  - parse success rate: 100%
  - thèmes dominants: support client insuffisant (36%), service technique (22%), tarification (15%)

## Sorties produites
Tous les livrables sont dans `output_script/`.

- Exploration:
  - `distribution_csat_globale.html`
  - `csat_moyen_par_offre.html`
  - `csat_par_offre.csv`
- NLP classique:
  - `top_termes.csv`, `top_termes.html`
  - `top_bigrams.csv`, `top_bigrams.html`
  - `top_trigrams.csv`
  - `tfidf_scores.csv`
- Baseline ML:
  - `baseline_metrics.csv`
  - `baseline_classification_report.csv`
  - `baseline_top_negative_terms.csv`
  - `baseline_top_positive_terms.csv`
- Topic modeling (LDA):
  - `lda_topics_summary.csv`
  - `lda_topics_detailed.csv`
  - `lda_topic_distribution.csv`
  - `lda_topics_visualization.html`
  - `verbatims_negatifs_avec_topics.csv`
- LLM:
  - `llm_enrichment_results.csv`
  - `llm_enrichment_results.jsonl`
- Synthèse:
  - `rapport_synthese.txt`

## Exécution
### 1) Lancer l'analyse complète
```bash
python analyse_csat_complete_standalone.py
```

### 2) Lancer le dashboard
```bash
streamlit run dashboard_llm_streamlit.py
```

## Prérequis
- Python 3.10+
- Dépendances principales: `pandas`, `numpy`, `scikit-learn`, `plotly`, `streamlit`
- Pour la partie LLM: `openai`, `python-dotenv`, variable `OPENAI_API_KEY`

## Positionnement entretien
Ce cas montre:
- une approche analytique structurée et industrialisable
- un arbitrage coût/valeur (NLP classique avant LLM)
- des livrables lisibles pour un usage métier (CSV, HTML, rapport, dashboard)
- une capacité à relier exploration, modélisation et recommandations opérationnelles
