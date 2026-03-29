#!/usr/bin/env python
# coding: utf-8

"""
===================================================================================
ANALYSE CSAT - Script complet et autonome pour entretien technique
===================================================================================

Ce script effectue une analyse complète des données CSAT (Customer Satisfaction)
en utilisant plusieurs approches complémentaires :

1. CHARGEMENT ET EXPLORATION DES DONNÉES
   - Lecture depuis une base SQLite
   - Analyse de la qualité des données (valeurs manquantes, cohérence)
   - Exploration des distributions CSAT/NPS par société et produit

2. ANALYSE NLP CLASSIQUE (Approche économique)
   - Extraction des verbatims négatifs (CSAT <= 3, NPS <= 6)
   - Bag of Words (BoW) et Term Frequency
   - Bigrams et Trigrams pour identifier les combinaisons fréquentes
   - TF-IDF pour pondérer l'importance des termes

3. MODÈLE BASELINE SUPERVISÉ
   - Classification négatif/positif avec Logistic Regression
   - TF-IDF comme features
   - Métriques de performance (F1, AUC, précision, rappel)
   - Identification des termes discriminants

4. ANALYSE THÉMATIQUE (LDA)
   - Latent Dirichlet Allocation pour découvrir les topics latents
   - Interprétation des thèmes récurrents dans les verbatims négatifs

5. ENRICHISSEMENT LLM
   - Classification thématique fine avec GPT-4
   - Évaluation de la sévérité et recommandations d'actions
   - Cache local pour minimiser les coûts API
   - Taxonomie fixe de 8 thèmes métier
   (Nécessite OPENAI_API_KEY - non bloquant si absente)

6. RAPPORT DE SYNTHÈSE
   - Consolidation de tous les résultats
   - Métriques clés et insights actionnables

OUTPUTS:
- Dossier output_script/ avec tous les CSV et graphiques générés
- Cache LLM persistant (outputs/llm_cache.jsonl)
- Rapports d'analyse intermédiaires

APPROCHE :
Le script privilégie les techniques NLP classiques (économiques et rapides)
avant d'envisager l'enrichissement LLM. Cette approche "Start simple, scale if needed"
permet d'extraire 80% des insights pour <1% du coût.

Auteur: Analyse CSAT
Date: Mars 2026
===================================================================================
"""

import os
import sys
import sqlite3
import warnings
import hashlib
import json
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import re

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

warnings.filterwarnings('ignore')

# ===================================================================================
# CONFIGURATION GLOBALE
# ===================================================================================

# Paramètres de l'analyse
CONFIG = {
    'db_path': 'db.sqlite',
    'table_name': 'csat_extract',
    'output_dir': 'output_script',
    'csat_threshold': 3,      # CSAT <= 3 considéré comme négatif
    'nps_threshold': 6,       # NPS <= 6 considéré comme négatif
    'top_n_terms': 30,        # Nombre de termes à extraire
    'lda_n_topics': 8,        # Nombre de topics pour LDA
    'lda_n_top_words': 12,    # Mots par topic
    'lda_min_df': 10,         # Fréquence minimale des mots
    'lda_max_df': 0.6,        # Fréquence maximale (filtrer mots trop communs)
    'random_state': 42,
    'llm_max_rows': 100,      # Nombre de verbatims à enrichir avec LLM (coût contrôlé)
    'llm_batch_size': 50,     # Taille des batchs pour l'API OpenAI
}

# Stopwords français étendus (incluant le bruit métier observé)
STOPWORDS_FR = {
    # Mots grammaticaux de base
    "de", "la", "le", "les", "un", "une", "des", "du", "d", "au", "aux", "et", "ou",
    "donc", "or", "ni", "car", "je", "tu", "il", "elle", "on", "nous", "vous", "ils",
    "elles", "me", "m", "te", "t", "se", "s", "ce", "cet", "cette", "ces", "ça", "ca",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "pas", "plus", "moins",
    "très", "tres", "tout", "tous", "toute", "toutes", "qui", "que", "qu", "quoi",
    "dont", "où", "a", "à", "est", "sont", "été", "etre", "être", "ai", "as", "avons",
    "avez", "ont", "fait", "faire", "comme", "mais", "si", "ne", "n", "y", "en", "l",
    "j", "c", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "leur",
    "leurs", "j'ai", "suis", "chez", "c'est", "depuis", "n'ai", "alors", "encore",
    "vraiment", "beaucoup", "peu", "toujours", "jamais", "aussi", "même", "autre",
    "cela", "celui", "celle", "ceux", "celles", "ici", "là", "quand", "comment",
    "pourquoi",
    # Bruit métier spécifique (ajusté après observations)
    "free", "service", "client", "conseiller", "votre", "vos", "bien", "dit", "vais",
    "avoir", "fois", "non", "m'a", "n'a", "n'est", "bonjour", "après", "moi",
    "plusieurs", "deux", "était", "aucun", "aucune", "rien", "personne", "merci",
    "part", "demandé", "qu'il", "jours", "mois", "numéro"
}

# Taxonomie fixe de thèmes (définie avec le métier)
DEFINED_THEMES = [
    "support client insuffisant",
    "problemes de carte sim",
    "probleme d'internet",
    "service technique",
    "probleme de communication",
    "probleme avec le produit",
    "tarification",
    "churn",
]


# ===================================================================================
# PARTIE 0: FONCTIONS UTILITAIRES LLM (AUTONOMES)
# ===================================================================================

@dataclass
class LLMConfig:
    """Configuration pour l'enrichissement LLM."""
    model: str = "gpt-4.1-mini"
    prompt_version: str = "v6-batch-fixed-themes-with-reasons"
    max_input_chars: int = 2000
    temperature: float = 0.0


def _to_jsonable(value: Any) -> Any:
    """Convertit une valeur Python en format JSON-serializable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _hash_record(text: str, prompt_version: str, constraint_signature: str = "") -> str:
    """
    Génère un hash SHA256 unique pour un verbatim.
    
    Utilisé pour le système de cache : si le hash existe déjà, on réutilise
    le résultat précédent sans faire d'appel API (économie de coût).
    """
    payload = f"{prompt_version}::{constraint_signature}::{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_cache(cache_path: Path) -> dict:
    """
    Lit le cache LLM depuis un fichier JSONL.
    
    Le cache est un dictionnaire {hash: résultat} qui permet de réutiliser
    les enrichissements précédents sans repayer l'API.
    """
    cache = {}
    if not cache_path.exists():
        return cache
    
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            key = obj.get("cache_key")
            if isinstance(key, str):
                cache[key] = obj
        except json.JSONDecodeError:
            continue
    return cache


def _append_cache(cache_path: Path, record: dict) -> None:
    """Ajoute un enregistrement au cache (format JSONL)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")


def _extract_json_list(text: str) -> Optional[List[dict]]:
    """
    Extrait une liste JSON de la réponse du LLM.
    
    Le LLM peut renvoyer du JSON dans différents formats :
    - Pur JSON: [{"key": "value"}, ...]
    - Markdown fenced: ```json\n[...]\n```
    - Enveloppé: {"results": [...]}
    
    Cette fonction gère tous ces cas.
    """
    text = text.strip()
    if not text:
        return None
    
    # Gérer les blocs markdown (```json ... ```)
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            text = text[first_newline + 1 : last_fence].strip()
    
    # Essayer de parser directement
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            # Chercher une liste dans les clés communes
            for key in ("results", "items", "data", "output", "predictions"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    
    # Chercher un tableau JSON dans le texte
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        maybe_json = text[start : end + 1]
        try:
            parsed = json.loads(maybe_json)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            return None
    
    return None


def _build_llm_batch_messages(batch: List[Tuple[Any, str]], 
                               allowed_themes: Optional[List[str]] = None) -> List[dict]:
    """
    Construit les messages pour l'API OpenAI en mode batch.
    
    Le batch processing permet d'envoyer plusieurs verbatims en une seule
    requête API, ce qui est plus efficace et économique que de traiter
    chaque verbatim individuellement.
    
    Args:
        batch: Liste de tuples (index, texte)
        allowed_themes: Liste des thèmes autorisés (taxonomie fixe)
        
    Returns:
        Messages formatés pour l'API OpenAI (system + user)
    """
    theme_constraint = ""
    if allowed_themes:
        theme_list = ", ".join(f'"{t}"' for t in allowed_themes)
        theme_constraint = (
            " Theme must be one of the following values exactly: "
            f"[{theme_list}]. If uncertain, use \"churn\"."
        )
    
    system_prompt = (
        "You are an analyst for French CSAT reviews. Return JSON only. In French. "
        "Output MUST be a JSON array where each element corresponds to one input item in the same order. "
        "Each element MUST include \"position\" (int) copied from input. "
        "Each element schema: {\"position\": int, \"theme\": str, \"sub_theme\": str (with brief reason explanation), \"severity\": int (1-5), "
        "\"main_issue\": str, \"recommended_action\": str, \"confidence\": float (0-1)}."
        + theme_constraint
    )
    
    # Préparer le payload avec position pour maintenir l'ordre
    payload = [{"position": i, "verbatim": text} for i, (_, text) in enumerate(batch)]
    payload_json = json.dumps(payload, ensure_ascii=False)
    
    user_prompt = (
        "Analyze each verbatim and return only a JSON array in the same order as input.\n"
        "No markdown, no prose.\n\n"
        "IMPORTANT: sub_theme MUST include a concise explanation of WHY this issue occurs or its underlying cause.\n"
        "Example sub_theme format: 'Routeur 4G defaillant (manque de couverture reseau)'\n\n"
        f"Input:\n{payload_json}"
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _chunk_list(items: List[Tuple], batch_size: int) -> List[List[Tuple]]:
    """Découpe une liste en chunks de taille batch_size."""
    if batch_size <= 0:
        batch_size = 1
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def run_llm_enrichment(reviews: pd.Series,
                      api_key: str,
                      config: Optional[LLMConfig] = None,
                      cache_path: str = "outputs/llm_cache.jsonl",
                      max_rows: Optional[int] = None,
                      batch_size: int = 50,
                      max_retries: int = 5,
                      retry_base_seconds: float = 5.0,
                      inter_batch_delay_seconds: float = 2.0,
                      allowed_themes: Optional[List[str]] = None,
                      source_df: Optional[pd.DataFrame] = None,
                      trace_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Enrichit les verbatims avec un LLM (GPT) pour classification thématique fine.
    
    Cette fonction est le cœur de l'enrichissement LLM. Elle :
    1. Vérifie le cache pour chaque verbatim (économie)
    2. Groupe les verbatims non-cachés en batchs
    3. Appelle l'API OpenAI avec retry/backoff
    4. Parse et valide les réponses JSON
    5. Sauvegarde les résultats dans le cache
    6. Retourne un DataFrame enrichi
    
    MÉCANISME DE BATCH :
    -------------------
    Au lieu d'envoyer 100 requêtes séparées (coûteux et lent), on groupe
    les verbatims par 50 et on envoie 2 requêtes. Le LLM traite le batch
    entier et retourne une liste ordonnée de résultats.
    
    ROBUSTESSE :
    -----------
    - Retry avec backoff exponentiel si erreur API
    - Gestion des timeouts et rate limits
    - Validation stricte du JSON retourné
    - Traçabilité complète (erreurs, timing, cache hits)
    
    Args:
        reviews: Série pandas contenant les verbatims
        api_key: Clé API OpenAI
        config: Configuration LLM (modèle, température, etc.)
        cache_path: Chemin vers le fichier de cache JSONL
        max_rows: Nombre maximum de verbatims à traiter (pour contrôler les coûts)
        batch_size: Nombre de verbatims par batch (50 recommandé)
        max_retries: Nombre de tentatives en cas d'erreur
        retry_base_seconds: Délai de base pour le backoff exponentiel
        inter_batch_delay_seconds: Délai entre les batchs (rate limiting)
        allowed_themes: Taxonomie fixe de thèmes (classification contrainte)
        source_df: DataFrame source pour traçabilité
        trace_columns: Colonnes à conserver pour la traçabilité
        
    Returns:
        DataFrame enrichi avec colonnes : theme, sub_theme, severity, main_issue,
        recommended_action, confidence, cache_hit, parse_ok, error, etc.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Le package 'openai' est requis. Installer avec: pip install openai")
    
    if config is None:
        config = LLMConfig()
    
    client = OpenAI(api_key=api_key)
    
    # Préparation des données
    reviews = reviews.fillna("").astype(str)
    reviews = reviews[reviews.str.strip().ne("")]
    if max_rows is not None:
        reviews = reviews.head(max_rows)
    
    print(f"   Verbatims à traiter : {len(reviews)}")
    
    # Chargement du cache
    cache_file = Path(cache_path)
    cache = _read_cache(cache_file)
    print(f"   Entrées en cache : {len(cache)}")
    
    # Signature de contrainte pour le cache (thèmes)
    constraint_signature = ""
    if allowed_themes:
        constraint_signature = "|".join(sorted([t.strip() for t in allowed_themes if isinstance(t, str)]))
    
    # Préparation de la traçabilité
    trace_columns_existing = []
    trace_lookup = None
    if source_df is not None:
        if trace_columns is None:
            trace_columns = ["contact_id", "id_profil", "id_conseiller", "date_contact"]
        trace_columns_existing = [c for c in trace_columns if c in source_df.columns]
        if len(trace_columns_existing) > 0:
            trace_lookup = source_df[trace_columns_existing].groupby(level=0, sort=False).first()
    
    def _trace_payload(idx: Any) -> dict:
        """Récupère les colonnes de traçabilité pour un index donné."""
        if trace_lookup is None:
            return {}
        if idx not in trace_lookup.index:
            return {col: None for col in trace_columns_existing}
        row = trace_lookup.loc[idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {col: row.get(col, None) for col in trace_columns_existing}
    
    # Phase 1: Vérifier le cache et identifier les verbatims à enrichir
    rows = []
    pending = []
    
    for idx, text in reviews.items():
        text_cut = text[:config.max_input_chars]
        key = _hash_record(text_cut, config.prompt_version, constraint_signature=constraint_signature)
        
        if key in cache:
            # Cache hit: réutiliser le résultat
            cached_record = cache[key]
            row_data = {
                "index": idx,
                "commentaire": text,
                "cache_hit": True,
                "parse_ok": cached_record.get("parse_ok", False),
                "error": cached_record.get("error"),
            }
            # Copier les champs LLM
            for field in ["theme", "sub_theme", "severity", "main_issue", "recommended_action", "confidence"]:
                row_data[field] = cached_record.get(field)
            # Ajouter la traçabilité
            row_data.update(_trace_payload(idx))
            rows.append(row_data)
        else:
            # Cache miss: ajouter à la liste des pendants
            pending.append((idx, text_cut))
    
    cache_hits = len(rows)
    print(f"   Cache hits : {cache_hits} ({cache_hits/len(reviews)*100:.1f}%)")
    print(f"   A enrichir : {len(pending)}")
    
    if len(pending) == 0:
        print("   Tous les verbatims sont en cache!")
        return pd.DataFrame(rows)
    
    # Phase 2: Traiter les verbatims en batchs
    batches = _chunk_list(pending, batch_size)
    print(f"   Nombre de batchs : {len(batches)}")
    print()
    
    for batch_idx, batch in enumerate(batches, 1):
        print(f"   Batch {batch_idx}/{len(batches)} ({len(batch)} verbatims)...", end=" ", flush=True)
        
        # Construire les messages pour l'API
        messages = _build_llm_batch_messages(batch, allowed_themes=allowed_themes)
        
        # Appel API avec retry
        response_items = None
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    timeout=60.0,
                )
                
                raw_content = response.choices[0].message.content
                response_items = _extract_json_list(raw_content)
                
                if response_items and len(response_items) == len(batch):
                    # Succès
                    print(f"OK (tentative {attempt})")
                    break
                else:
                    # Format invalide
                    last_error = f"Invalid JSON format or length mismatch (got {len(response_items or [])} items, expected {len(batch)})"
                    if attempt < max_retries:
                        wait = retry_base_seconds * (2 ** (attempt - 1))
                        print(f"Retry {attempt}/{max_retries} (wait {wait}s)...", end=" ", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"FAILED after {max_retries} attempts")
            
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    wait = retry_base_seconds * (2 ** (attempt - 1))
                    print(f"Error: {last_error[:50]}... Retry {attempt}/{max_retries} (wait {wait}s)...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"FAILED: {last_error[:100]}")
        
        # Traiter les résultats du batch
        if response_items and len(response_items) == len(batch):
            for (idx, text_cut), item in zip(batch, response_items):
                row_data = {
                    "index": idx,
                    "commentaire": reviews.loc[idx],
                    "cache_hit": False,
                    "parse_ok": True,
                    "error": None,
                    "theme": item.get("theme"),
                    "sub_theme": item.get("sub_theme"),
                    "severity": item.get("severity"),
                    "main_issue": item.get("main_issue"),
                    "recommended_action": item.get("recommended_action"),
                    "confidence": item.get("confidence"),
                }
                row_data.update(_trace_payload(idx))
                rows.append(row_data)
                
                # Sauvegarder dans le cache
                key = _hash_record(text_cut, config.prompt_version, constraint_signature=constraint_signature)
                cache_record = {
                    "cache_key": key,
                    "text": text_cut,
                    "prompt_version": config.prompt_version,
                    "constraint_signature": constraint_signature,
                    "parse_ok": True,
                    "error": None,
                    **{k: row_data[k] for k in ["theme", "sub_theme", "severity", "main_issue", "recommended_action", "confidence"]}
                }
                _append_cache(cache_file, cache_record)
        else:
            # Échec du batch: enregistrer l'erreur pour chaque verbatim
            for idx, text_cut in batch:
                row_data = {
                    "index": idx,
                    "commentaire": reviews.loc[idx],
                    "cache_hit": False,
                    "parse_ok": False,
                    "error": last_error,
                    "theme": None,
                    "sub_theme": None,
                    "severity": None,
                    "main_issue": None,
                    "recommended_action": None,
                    "confidence": None,
                }
                row_data.update(_trace_payload(idx))
                rows.append(row_data)
                
                # Sauvegarder l'erreur dans le cache
                key = _hash_record(text_cut, config.prompt_version, constraint_signature=constraint_signature)
                cache_record = {
                    "cache_key": key,
                    "text": text_cut,
                    "prompt_version": config.prompt_version,
                    "constraint_signature": constraint_signature,
                    "parse_ok": False,
                    "error": last_error,
                }
                _append_cache(cache_file, cache_record)
        
        # Délai inter-batch pour rate limiting
        if batch_idx < len(batches) and inter_batch_delay_seconds > 0:
            time.sleep(inter_batch_delay_seconds)
    
    print()
    return pd.DataFrame(rows)


# ===================================================================================
# PARTIE 1: CHARGEMENT ET EXPLORATION DES DONNÉES
# ===================================================================================

def load_data_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Charge les données depuis une base SQLite.
    
    Args:
        db_path: Chemin vers le fichier SQLite
        table_name: Nom de la table à charger
        
    Returns:
        DataFrame pandas avec les données CSAT
    """
    print(f"[DATA] Chargement des données depuis {db_path}...")
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"[OK] {len(df)} lignes chargées avec {len(df.columns)} colonnes\n")
    return df


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Analyse la qualité des données : valeurs manquantes, types, cohérence.
    
    Cette étape est cruciale pour comprendre les limites de l'analyse
    et identifier les biais potentiels.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dictionnaire avec les métriques de qualité
    """
    print("[CHECK] Analyse de la qualité des données...")
    
    # Conversion NPS en numérique (peut contenir '\N')
    df['nps_num'] = pd.to_numeric(df['nps'], errors='coerce')
    df['csat_num'] = pd.to_numeric(df['csat'], errors='coerce')
    
    quality_report = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_by_column': df.isnull().sum(),
        'empty_comments': df['commentaire'].isna().sum(),
        'invalid_nps': df['nps_num'].isna().sum(),
        'invalid_csat': df['csat_num'].isna().sum(),
        'csat_range': (df['csat_num'].min(), df['csat_num'].max()),
        'nps_range': (df['nps_num'].min(), df['nps_num'].max()),
    }
    
    print(f"  - Lignes: {quality_report['row_count']}")
    print(f"  - Colonnes: {quality_report['column_count']}")
    print(f"  - Commentaires vides: {quality_report['empty_comments']}")
    print(f"  - NPS invalides: {quality_report['invalid_nps']}")
    print(f"  - CSAT invalides: {quality_report['invalid_csat']}")
    print(f"  - Range CSAT: {quality_report['csat_range']}")
    print(f"  - Range NPS: {quality_report['nps_range']}\n")
    
    return quality_report


def explore_distributions(df: pd.DataFrame, output_dir: str):
    """
    Explore les distributions CSAT/NPS par société et par produit.
    
    Cette analyse permet d'identifier les segments à problème et de contextualiser
    les verbatims négatifs par rapport à la performance globale.
    """
    print("[GRAPH] Exploration des distributions CSAT/NPS...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Préparation des données
    df_plot = df.copy()
    df_plot['offer_label_clean'] = df_plot['offer_label'].fillna('SANS PRODUIT')
    
    # 1. Distribution CSAT globale
    fig_csat = px.histogram(
        df_plot, x='csat',
        title='Distribution globale du CSAT',
        labels={'csat': 'Score CSAT', 'count': 'Nombre de commentaires'}
    )
    fig_csat.write_html(f'{output_dir}/distribution_csat_globale.html')
    print(f"  + Graphique sauvegardé: distribution_csat_globale.html")
    
    # 2. CSAT moyen par offre
    csat_by_offer = df_plot.groupby('offer_label_clean').agg({
        'csat': ['mean', 'median', 'count']
    }).reset_index()
    csat_by_offer.columns = ['offre', 'csat_moyen', 'csat_median', 'nb_commentaires']
    csat_by_offer = csat_by_offer.sort_values('csat_moyen', ascending=False)
    
    fig_offer = go.Figure(data=[go.Bar(
        x=csat_by_offer['csat_moyen'],
        y=csat_by_offer['offre'],
        text=csat_by_offer['nb_commentaires'],
        textposition='auto',
        orientation='h',
        marker=dict(
            color=csat_by_offer['nb_commentaires'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Nb comments")
        )
    )])
    fig_offer.update_layout(
        title="CSAT moyen par offre",
        xaxis_title="CSAT moyen",
        yaxis_title="Offre",
        height=600,
        width=1000
    )
    fig_offer.write_html(f'{output_dir}/csat_moyen_par_offre.html')
    print(f"  + Graphique sauvegardé: csat_moyen_par_offre.html")
    
    # Sauvegarde CSV
    csat_by_offer.to_csv(f'{output_dir}/csat_par_offre.csv', index=False, encoding='utf-8-sig')
    print(f"  + CSV sauvegardé: csat_par_offre.csv\n")
    
    return csat_by_offer


# ===================================================================================
# PARTIE 2: ANALYSE NLP CLASSIQUE (Approche économique)
# ===================================================================================

def tokenize(text: str) -> List[str]:
    """
    Tokenize et nettoie un texte en français.
    
    - Conversion en minuscules
    - Suppression de la ponctuation et caractères spéciaux
    - Filtrage des stopwords et mots courts (< 3 caractères)
    """
    text = text.lower()
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüÿçœæ'-]", " ", text)
    tokens = re.findall(r"[a-zàâäéèêëîïôöùûüÿçœæ']+", text)
    return [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS_FR]


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Extrait les n-grams d'une liste de tokens.
    
    Args:
        tokens: Liste de tokens
        n: Taille des n-grams (2 pour bigrams, 3 pour trigrams)
    """
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_negative_reviews(df: pd.DataFrame, 
                             csat_threshold: int = 3, 
                             nps_threshold: int = 6) -> pd.DataFrame:
    """
    Extrait les verbatims négatifs selon les seuils CSAT et NPS.
    
    Définition de "négatif" : CSAT <= 3 ET NPS <= 6
    Ces seuils sont ajustables selon la définition métier.
    
    Args:
        df: DataFrame complet
        csat_threshold: Seuil CSAT (<= négatif)
        nps_threshold: Seuil NPS (<= négatif)
        
    Returns:
        DataFrame filtré sur les verbatims négatifs
    """
    print(f"[NEG] Extraction des verbatims négatifs (CSAT <= {csat_threshold}, NPS <= {nps_threshold})...")
    
    df_work = df.copy()
    df_work['nps_num'] = pd.to_numeric(df_work['nps'], errors='coerce')
    df_work['csat_num'] = pd.to_numeric(df_work['csat'], errors='coerce')
    
    mask = (
        (df_work['csat_num'] <= csat_threshold) &
        (df_work['nps_num'] <= nps_threshold) &
        (df_work['commentaire'].notna())
    )
    
    neg_df = df_work[mask].copy()
    print(f"[OK] {len(neg_df)} verbatims négatifs extraits ({len(neg_df)/len(df)*100:.1f}% du total)\n")
    
    return neg_df


def analyze_classical_nlp(neg_df: pd.DataFrame, 
                          top_n: int = 30, 
                          output_dir: str = 'output_script') -> Dict:
    """
    Analyse NLP classique : unigrammes, bigrams, trigrams, BoW, TF-IDF.
    
    Cette approche permet d'identifier rapidement les termes et expressions
    problématiques sans recourir à des modèles coûteux. C'est une première étape
    essentielle avant d'envisager des enrichissements LLM.
    
    Args:
        neg_df: DataFrame des verbatims négatifs
        top_n: Nombre de termes/n-grams à extraire
        output_dir: Dossier de sortie
        
    Returns:
        Dictionnaire avec tous les résultats NLP
    """
    print(f"[NLP] Analyse NLP classique sur {len(neg_df)} verbatims négatifs...")
    
    reviews = neg_df['commentaire'].dropna().astype(str)
    
    # 1. UNIGRAMMES (mots simples)
    counter = Counter()
    for txt in reviews:
        counter.update(tokenize(txt))
    
    top_terms = pd.DataFrame(counter.most_common(top_n), columns=['terme', 'frequence'])
    
    # 2. BIGRAMS (2 mots consécutifs)
    bigram_counter = Counter()
    for txt in reviews:
        toks = tokenize(txt)
        bigram_counter.update(extract_ngrams(toks, n=2))
    
    top_bigrams = pd.DataFrame(bigram_counter.most_common(top_n), columns=['bigram', 'frequence'])
    
    # 3. TRIGRAMS (3 mots consécutifs)
    trigram_counter = Counter()
    for txt in reviews:
        toks = tokenize(txt)
        trigram_counter.update(extract_ngrams(toks, n=3))
    
    top_trigrams = pd.DataFrame(trigram_counter.most_common(top_n), columns=['trigram', 'frequence'])
    
    # 4. BAG OF WORDS (matrice de fréquences)
    vocab = [mot for mot, _ in counter.most_common(top_n)]
    bow = pd.DataFrame(0, index=reviews.index, columns=vocab)
    
    for idx, txt in reviews.items():
        toks = tokenize(txt)
        c = Counter(toks)
        for mot in vocab:
            bow.at[idx, mot] = c.get(mot, 0)
    
    # 5. TF-IDF (importance pondérée des termes)
    if len(vocab) > 0 and len(reviews) > 0:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            vocabulary=vocab
        )
        tfidf_matrix = vectorizer.fit_transform(reviews)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=reviews.index,
            columns=vectorizer.get_feature_names_out()
        )
        tfidf_scores = tfidf_df.mean().sort_values(ascending=False).reset_index()
        tfidf_scores.columns = ['terme', 'score_tfidf_moyen']
    else:
        tfidf_df = pd.DataFrame()
        tfidf_scores = pd.DataFrame()
    
    # Sauvegarde des résultats
    os.makedirs(output_dir, exist_ok=True)
    top_terms.to_csv(f'{output_dir}/top_termes.csv', index=False, encoding='utf-8-sig')
    top_bigrams.to_csv(f'{output_dir}/top_bigrams.csv', index=False, encoding='utf-8-sig')
    top_trigrams.to_csv(f'{output_dir}/top_trigrams.csv', index=False, encoding='utf-8-sig')
    
    if not tfidf_scores.empty:
        tfidf_scores.to_csv(f'{output_dir}/tfidf_scores.csv', index=False, encoding='utf-8-sig')
    
    print(f"  + CSVs sauvegardés: top_termes.csv, top_bigrams.csv, top_trigrams.csv")
    
    # Visualisations
    if not top_terms.empty:
        fig_terms = px.bar(
            top_terms.sort_values('frequence'),
            x='frequence', y='terme', orientation='h',
            title=f'Top {top_n} termes - Verbatims négatifs'
        )
        fig_terms.update_layout(height=700, width=1000)
        fig_terms.write_html(f'{output_dir}/top_termes.html')
        print(f"  + Graphique sauvegardé: top_termes.html")
    
    if not top_bigrams.empty:
        fig_bigrams = px.bar(
            top_bigrams.sort_values('frequence'),
            x='frequence', y='bigram', orientation='h',
            title=f'Top {top_n} bigrams - Verbatims négatifs'
        )
        fig_bigrams.update_layout(height=700, width=1000)
        fig_bigrams.write_html(f'{output_dir}/top_bigrams.html')
        print(f"  + Graphique sauvegardé: top_bigrams.html")
    
    print()
    
    return {
        'top_terms': top_terms,
        'top_bigrams': top_bigrams,
        'top_trigrams': top_trigrams,
        'bow': bow,
        'tfidf': tfidf_df,
        'tfidf_scores': tfidf_scores
    }


# ===================================================================================
# PARTIE 3: MODÈLE BASELINE SUPERVISÉ
# ===================================================================================

def train_baseline_classifier(df: pd.DataFrame, 
                              csat_threshold: int = 3,
                              nps_threshold: int = 6,
                              output_dir: str = 'output_script') -> Dict:
    """
    Entraîne un modèle baseline de classification négatif/positif.
    
    Approche : TF-IDF + Logistic Regression
    
    Objectifs :
    1. Évaluer la séparabilité des classes (négatif vs positif)
    2. Identifier les features discriminantes (mots clés)
    3. Établir une baseline de performance pour de futurs modèles
    
    Args:
        df: DataFrame complet
        csat_threshold: Seuil CSAT pour définir "négatif"
        nps_threshold: Seuil NPS pour définir "négatif"
        output_dir: Dossier de sortie
        
    Returns:
        Dictionnaire avec métriques et résultats
    """
    print(f"[ML] Entraînement du modèle baseline (TF-IDF + Logistic Regression)...")
    
    df_work = df.copy()
    df_work['nps_num'] = pd.to_numeric(df_work['nps'], errors='coerce')
    df_work['csat_num'] = pd.to_numeric(df_work['csat'], errors='coerce')
    
    # Création du label binaire (négatif = 1, positif = 0)
    df_work['label'] = (
        (df_work['csat_num'] <= csat_threshold) & 
        (df_work['nps_num'] <= nps_threshold)
    ).astype(int)
    
    # Filtrage des commentaires valides
    df_model = df_work[df_work['commentaire'].notna()].copy()
    
    X = df_model['commentaire'].astype(str)
    y = df_model['label']
    
    print(f"  - Distribution des classes: {y.value_counts().to_dict()}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['random_state'], stratify=y
    )
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=500,
        min_df=5,
        max_df=0.7
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Entraînement du modèle
    clf = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'], class_weight='balanced')
    clf.fit(X_train_tfidf, y_train)
    
    # Prédictions et métriques
    y_pred = clf.predict(X_test_tfidf)
    y_pred_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n  [METRICS] Performance du modèle:")
    print(f"    - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    - Precision: {metrics['precision']:.4f}")
    print(f"    - Recall:    {metrics['recall']:.4f}")
    print(f"    - F1 Score:  {metrics['f1']:.4f}")
    print(f"    - AUC:       {metrics['auc']:.4f}")
    
    # Classification report détaillé
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Identification des termes les plus discriminants
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    
    # Top termes négatifs (coef positifs car label=1 pour négatif)
    top_neg_idx = coef.argsort()[-30:][::-1]
    top_negative_terms = pd.DataFrame({
        'terme': [feature_names[i] for i in top_neg_idx],
        'coefficient': [coef[i] for i in top_neg_idx]
    })
    
    # Top termes positifs (coef négatifs)
    top_pos_idx = coef.argsort()[:30]
    top_positive_terms = pd.DataFrame({
        'terme': [feature_names[i] for i in top_pos_idx],
        'coefficient': [coef[i] for i in top_pos_idx]
    })
    
    # Sauvegarde des résultats
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{output_dir}/baseline_metrics.csv', index=False)
    
    class_report_df.to_csv(f'{output_dir}/baseline_classification_report.csv', encoding='utf-8-sig')
    top_negative_terms.to_csv(f'{output_dir}/baseline_top_negative_terms.csv', index=False, encoding='utf-8-sig')
    top_positive_terms.to_csv(f'{output_dir}/baseline_top_positive_terms.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n  + Résultats sauvegardés dans {output_dir}/")
    print()
    
    return {
        'metrics': metrics,
        'classification_report': class_report,
        'top_negative_terms': top_negative_terms,
        'top_positive_terms': top_positive_terms,
        'model': clf,
        'vectorizer': vectorizer
    }


# ===================================================================================
# PARTIE 4: ANALYSE THÉMATIQUE (LDA)
# ===================================================================================

def run_lda_topic_modeling(neg_df: pd.DataFrame,
                          n_topics: int = 8,
                          n_top_words: int = 12,
                          min_df: int = 10,
                          max_df: float = 0.6,
                          output_dir: str = 'output_script') -> Dict:
    """
    Effectue une analyse LDA (Latent Dirichlet Allocation) pour découvrir
    les thèmes latents dans les verbatims négatifs.
    
    LDA est une technique de modélisation de topics qui identifie des groupes
    de mots co-occurents. Chaque topic représente un thème sous-jacent.
    
    Avantages de LDA :
    - Non supervisé (pas besoin de labels)
    - Interprétabilité (top mots par topic)
    - Révèle des patterns cachés
    
    Args:
        neg_df: DataFrame des verbatims négatifs
        n_topics: Nombre de topics à extraire
        n_top_words: Nombre de mots à afficher par topic
        min_df: Fréquence minimale d'un mot
        max_df: Fréquence maximale d'un mot (filtrer mots trop communs)
        output_dir: Dossier de sortie
        
    Returns:
        Dictionnaire avec les topics et leurs mots associés
    """
    print(f"[LDA] Analyse LDA : extraction de {n_topics} topics thématiques...")
    
    corpus = neg_df['commentaire'].dropna().astype(str)
    print(f"  - Corpus: {len(corpus)} documents")
    
    # Vectorisation avec CountVectorizer
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df
    )
    
    dtm = vectorizer.fit_transform(corpus)
    print(f"  - Vocabulaire: {len(vectorizer.get_feature_names_out())} termes uniques")
    
    # Entraînement du modèle LDA
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=CONFIG['random_state'],
        learning_method='batch',
        max_iter=20,
        n_jobs=-1
    )
    
    doc_topic = lda_model.fit_transform(dtm)
    
    # Calcul de la perplexité (métrique de qualité du modèle)
    perplexity = lda_model.perplexity(dtm)
    print(f"  - Perplexité du modèle: {perplexity:.2f}")
    
    # Extraction des top mots par topic
    feature_names = vectorizer.get_feature_names_out()
    
    topics_long_rows = []
    for topic_idx, topic_weights in enumerate(lda_model.components_):
        top_idx = topic_weights.argsort()[::-1][:n_top_words]
        for rank, i in enumerate(top_idx, start=1):
            topics_long_rows.append({
                'topic_id': topic_idx,
                'rank': rank,
                'word': feature_names[i],
                'weight': topic_weights[i]
            })
    
    topics_long_df = pd.DataFrame(topics_long_rows)
    
    # Tableau résumé (mots concaténés par topic)
    topics_summary = (
        topics_long_df.sort_values(['topic_id', 'rank'])
        .groupby('topic_id')['word']
        .apply(lambda s: ' | '.join(s))
        .reset_index(name='top_words')
    )
    
    # Attribution du topic dominant à chaque document
    dominant_topic = doc_topic.argmax(axis=1)
    neg_df_with_topic = neg_df.copy()
    neg_df_with_topic['dominant_topic'] = dominant_topic
    neg_df_with_topic['topic_probability'] = doc_topic.max(axis=1)
    
    # Statistiques par topic
    topic_distribution = pd.DataFrame({
        'topic_id': range(n_topics),
        'nb_documents': [(dominant_topic == i).sum() for i in range(n_topics)]
    })
    topic_distribution = topic_distribution.merge(topics_summary, on='topic_id')
    
    # Sauvegarde des résultats
    os.makedirs(output_dir, exist_ok=True)
    
    topics_summary.to_csv(f'{output_dir}/lda_topics_summary.csv', index=False, encoding='utf-8-sig')
    topics_long_df.to_csv(f'{output_dir}/lda_topics_detailed.csv', index=False, encoding='utf-8-sig')
    topic_distribution.to_csv(f'{output_dir}/lda_topic_distribution.csv', index=False, encoding='utf-8-sig')
    neg_df_with_topic.to_csv(f'{output_dir}/verbatims_negatifs_avec_topics.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n  + Topics extraits et sauvegardés:")
    print(f"    - lda_topics_summary.csv")
    print(f"    - lda_topics_detailed.csv")
    print(f"    - lda_topic_distribution.csv")
    print(f"    - verbatims_negatifs_avec_topics.csv")
    
    # Visualisation
    plot_df = topics_long_df.copy()
    plot_df['word_rank'] = plot_df['rank'].astype(str).str.zfill(2) + '. ' + plot_df['word']
    
    fig_topics = px.bar(
        plot_df.sort_values(['topic_id', 'rank'], ascending=[True, False]),
        x='weight',
        y='word_rank',
        facet_col='topic_id',
        facet_col_wrap=4,
        orientation='h',
        title=f'LDA - Top {n_top_words} mots par topic',
        height=900,
        width=1200
    )
    fig_topics.update_layout(showlegend=False)
    fig_topics.update_yaxes(title='')
    fig_topics.write_html(f'{output_dir}/lda_topics_visualization.html')
    print(f"    - lda_topics_visualization.html\n")
    
    return {
        'topics_summary': topics_summary,
        'topics_long': topics_long_df,
        'topic_distribution': topic_distribution,
        'perplexity': perplexity,
        'model': lda_model,
        'vectorizer': vectorizer,
        'df_with_topics': neg_df_with_topic
    }


# ===================================================================================
# PARTIE 5: ENRICHISSEMENT LLM (OPTIONNEL)
# ===================================================================================

def run_llm_enrichment_optional(neg_df: pd.DataFrame,
                                output_dir: str = 'output_script',
                                max_rows: Optional[int] = None) -> Optional[Dict]:
    """
    Enrichit les verbatims négatifs avec un LLM (GPT) pour classification thématique fine.
    
    SECTION OPTIONNELLE - Nécessite une clé API OpenAI valide
    
    Cette étape est facultative mais apporte une valeur ajoutée significative :
    - Classification automatique par thème (taxonomie fixe)
    - Évaluation de la sévérité (1-5)
    - Identification du problème principal
    - Recommandation d'action
    - Score de confiance
    
    POURQUOI APRÈS LE NLP CLASSIQUE ?
    ---------------------------------
    1. Coût/Bénéfice : Le NLP classique a déjà extrait 80% des insights pour <1% du coût
    2. Validation : On peut comparer les thèmes LLM aux topics LDA
    3. Efficacité : On enrichit seulement un sous-ensemble ciblé (ex: 100-500 verbatims)
    4. Cache local : Évite les appels redondants (coût maîtrisé)
    
    Args:
        neg_df: DataFrame des verbatims négatifs
        output_dir: Dossier de sortie
        max_rows: Nombre maximum de verbatims à enrichir (coût contrôlé)
        
    Returns:
        Dictionnaire avec résultats enrichis ou None si pas de clé API
    """
    print("[LLM] Enrichissement LLM : classification thématique fine avec GPT...")
    print("[WARN] Section OPTIONNELLE - Nécessite OPENAI_API_KEY dans .env\n")
    
    if max_rows is None:
        max_rows = CONFIG['llm_max_rows']
    
    # Vérification de la disponibilité de la clé API
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv('.env', usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
            print(f"[OK] Fichier .env trouvé: {env_path}")
        
        api_key = os.getenv('OPENAI_API_KEY', '').strip()
        
        if not api_key:
            print("[WARN] OPENAI_API_KEY non trouvée")
            print("   Cette étape est sautée (non bloquant)")
            print("   Le NLP classique et LDA ont déjà fourni les insights principaux\n")
            return None
            
        print(f"[OK] Clé API détectée (longueur: {len(api_key)} caractères)")
        
    except ImportError:
        print("[WARN] Package 'python-dotenv' non installé")
        print("   Cette étape est sautée (non bloquant)\n")
        return None
    
    # Import d'OpenAI
    try:
        from openai import OpenAI
        print("[OK] Package OpenAI disponible\n")
    except ImportError:
        print("[WARN] Package 'openai' non installé")
        print("   Installation: pip install openai")
        print("   Cette étape est sautée (non bloquant)\n")
        return None
    
    # Configuration de l'enrichissement LLM
    print(f"[CONFIG] Configuration de l'enrichissement:")
    print(f"   - Verbatims à enrichir: {min(max_rows, len(neg_df))}")
    print(f"   - Modèle: gpt-4.1-mini (économique et performant)")
    print(f"   - Cache: outputs/llm_cache.jsonl (réutilisation)")
    print(f"   - Batch size: {CONFIG['llm_batch_size']} verbatims par appel (efficacité)")
    print(f"   - Max retries: 6 avec backoff exponentiel")
    print(f"   - Inter-batch delay: 2 secondes (rate limiting)\n")
    
    print(f"[TAGS] Taxonomie fixe ({len(DEFINED_THEMES)} thèmes métier):")
    for i, theme in enumerate(DEFINED_THEMES, 1):
        print(f"   {i}. {theme}")
    print()
    
    # Préparation des colonnes de traçabilité
    trace_aliases = {
        'contact_id': ['contact_id', 'id_contact'],
        'id_profil': ['id_profil', 'profil_id', 'profile_id'],
        'id_conseiller': ['id_conseiller', 'conseiller_id', 'advisor_id'],
        'date_contact': ['date_contact', 'contact_date'],
    }
    
    neg_df_trace = neg_df.copy()
    for target_col, aliases in trace_aliases.items():
        src_col = next((c for c in aliases if c in neg_df_trace.columns), None)
        if src_col is None:
            neg_df_trace[target_col] = None
        else:
            neg_df_trace[target_col] = neg_df_trace[src_col]
    
    trace_cols = list(trace_aliases.keys())
    print(f"[LINK] Colonnes de traçabilité: {', '.join(trace_cols)}\n")
    
    try:
        # Configuration du LLM
        llm_cfg = LLMConfig(
            model='gpt-4.1-mini',
            prompt_version='v6-batch-fixed-themes-with-reasons'
        )
        
        print("[RUN] Lancement de l'enrichissement LLM...\n")
        print("[WAIT] Cela peut prendre 1-3 minutes selon le nombre de verbatims...")
        print("   (Les résultats en cache sont instantanés)\n")
        
        # Appel à la fonction d'enrichissement
        llm_df = run_llm_enrichment(
            reviews=neg_df_trace['commentaire'],
            api_key=api_key,
            config=llm_cfg,
            cache_path='outputs/llm_cache.jsonl',
            max_rows=max_rows,
            batch_size=CONFIG['llm_batch_size'],
            max_retries=6,
            retry_base_seconds=5.0,
            inter_batch_delay_seconds=2.0,
            allowed_themes=DEFINED_THEMES,
            source_df=neg_df_trace,
            trace_columns=trace_cols,
        )
        
        # Analyse des résultats
        print(f"\n[OK] Enrichissement terminé!")
        print(f"   - Lignes enrichies: {len(llm_df)}")
        
        if 'cache_hit' in llm_df.columns:
            cache_hit_rate = llm_df['cache_hit'].mean() * 100
            print(f"   - Cache hit rate: {cache_hit_rate:.1f}% (économies de coût)")
        
        if 'parse_ok' in llm_df.columns:
            parse_rate = llm_df['parse_ok'].mean() * 100
            print(f"   - Parse success rate: {parse_rate:.1f}%")
        
        # Statistiques sur les thèmes identifiés
        if 'theme' in llm_df.columns:
            print(f"\n[STATS] Distribution des thèmes:")
            theme_counts = llm_df['theme'].value_counts()
            for theme, count in theme_counts.head(8).items():
                pct = (count / len(llm_df)) * 100
                print(f"   * {theme}: {count} ({pct:.1f}%)")
        
        # Statistiques sur la sévérité
        if 'severity' in llm_df.columns:
            # Filtrer les valeurs non-null pour les statistiques
            severity_valid = llm_df['severity'].dropna()
            if len(severity_valid) > 0:
                severity_stats = severity_valid.describe()
                print(f"\n[STATS] Sévérité (échelle 1-5):")
                print(f"   - Moyenne: {severity_stats.loc['mean']:.2f}")
                print(f"   - Médiane: {severity_stats.loc['50%']:.0f}")
                print(f"   - Min/Max: {severity_stats.loc['min']:.0f} / {severity_stats.loc['max']:.0f}")
        
        # Sauvegarde des résultats
        os.makedirs(output_dir, exist_ok=True)
        
        # Export JSON Lines (format compact et facilement lisible)
        llm_df.to_json(
            f'{output_dir}/llm_enrichment_results.jsonl',
            orient='records',
            lines=True,
            force_ascii=False
        )
        print(f"\n[SAVE] Résultats sauvegardés:")
        print(f"   - {output_dir}/llm_enrichment_results.jsonl")
        
        # Export CSV pour Excel
        llm_df.to_csv(
            f'{output_dir}/llm_enrichment_results.csv',
            index=False,
            encoding='utf-8-sig'
        )
        print(f"   - {output_dir}/llm_enrichment_results.csv\n")
        
        return {
            'llm_df': llm_df,
            'cache_hit_rate': cache_hit_rate if 'cache_hit' in llm_df.columns else None,
            'parse_rate': parse_rate if 'parse_ok' in llm_df.columns else None,
            'theme_distribution': theme_counts if 'theme' in llm_df.columns else None
        }
        
    except Exception as e:
        print(f"\n[ERROR] Erreur lors de l'enrichissement LLM: {str(e)}")
        print("   Cette étape est sautée (non bloquant)")
        print("   Les analyses NLP classique et LDA restent valides\n")
        return None


# ===================================================================================
# PARTIE 6: SYNTHÈSE ET RAPPORTS
# ===================================================================================

def generate_summary_report(data_quality: Dict,
                           csat_by_offer: pd.DataFrame,
                           nlp_results: Dict,
                           baseline_results: Dict,
                           lda_results: Dict,
                           output_dir: str = 'output_script',
                           llm_results: Optional[Dict] = None):
    """
    Génère un rapport de synthèse en format texte et CSV.
    
    Ce rapport consolide tous les résultats et permet une lecture rapide
    des insights principaux pour l'entretien technique.
    """
    print("[REPORT] Génération du rapport de synthèse...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Création du rapport texte
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAPPORT D'ANALYSE CSAT - SYNTHÈSE")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Qualité des données
    report_lines.append("1. QUALITÉ DES DONNÉES")
    report_lines.append("-" * 80)
    report_lines.append(f"   - Nombre total de lignes: {data_quality['row_count']}")
    report_lines.append(f"   - Commentaires vides: {data_quality['empty_comments']} ({data_quality['empty_comments']/data_quality['row_count']*100:.1f}%)")
    report_lines.append(f"   - NPS invalides: {data_quality['invalid_nps']}")
    report_lines.append(f"   - CSAT invalides: {data_quality['invalid_csat']}")
    report_lines.append("")
    
    # 2. Distribution CSAT
    report_lines.append("2. DISTRIBUTION CSAT PAR OFFRE")
    report_lines.append("-" * 80)
    report_lines.append(f"   Top 5 offres avec le CSAT le plus élevé:")
    for idx, row in csat_by_offer.head(5).iterrows():
        report_lines.append(f"     * {row['offre'][:50]}: {row['csat_moyen']:.2f} ({row['nb_commentaires']} commentaires)")
    report_lines.append("")
    report_lines.append(f"   Top 5 offres avec le CSAT le plus faible:")
    for idx, row in csat_by_offer.tail(5).iterrows():
        report_lines.append(f"     * {row['offre'][:50]}: {row['csat_moyen']:.2f} ({row['nb_commentaires']} commentaires)")
    report_lines.append("")
    
    # 3. Termes clés négatifs
    report_lines.append("3. TERMES CLÉS DANS LES VERBATIMS NÉGATIFS")
    report_lines.append("-" * 80)
    report_lines.append(f"   Top 10 termes:")
    for idx, row in nlp_results['top_terms'].head(10).iterrows():
        report_lines.append(f"     * {row['terme']}: {row['frequence']} occurrences")
    report_lines.append("")
    report_lines.append(f"   Top 10 bigrams:")
    for idx, row in nlp_results['top_bigrams'].head(10).iterrows():
        report_lines.append(f"     * {row['bigram']}: {row['frequence']} occurrences")
    report_lines.append("")
    
    # 4. Performance du modèle baseline
    report_lines.append("4. MODÈLE BASELINE (CLASSIFICATION NÉGATIF/POSITIF)")
    report_lines.append("-" * 80)
    report_lines.append(f"   - Accuracy:  {baseline_results['metrics']['accuracy']:.4f}")
    report_lines.append(f"   - Precision: {baseline_results['metrics']['precision']:.4f}")
    report_lines.append(f"   - Recall:    {baseline_results['metrics']['recall']:.4f}")
    report_lines.append(f"   - F1 Score:  {baseline_results['metrics']['f1']:.4f}")
    report_lines.append(f"   - AUC:       {baseline_results['metrics']['auc']:.4f}")
    report_lines.append("")
    report_lines.append(f"   Top 5 termes discriminants (négatifs):")
    for idx, row in baseline_results['top_negative_terms'].head(5).iterrows():
        report_lines.append(f"     * {row['terme']}: {row['coefficient']:.4f}")
    report_lines.append("")
    
    # 5. Topics LDA
    report_lines.append("5. TOPICS THÉMATIQUES (LDA)")
    report_lines.append("-" * 80)
    report_lines.append(f"   Perplexité: {lda_results['perplexity']:.2f}")
    report_lines.append(f"   {len(lda_results['topics_summary'])} topics identifiés:")
    report_lines.append("")
    for idx, row in lda_results['topics_summary'].iterrows():
        nb_docs = lda_results['topic_distribution'].loc[
            lda_results['topic_distribution']['topic_id'] == row['topic_id'], 
            'nb_documents'
        ].values[0]
        report_lines.append(f"   Topic {row['topic_id']} ({nb_docs} documents):")
        report_lines.append(f"     {row['top_words']}")
        report_lines.append("")
    
    # 6. Enrichissement LLM (si disponible)
    if llm_results is not None and 'llm_df' in llm_results:
        report_lines.append("6. ENRICHISSEMENT LLM (GPT-4)")
        report_lines.append("-" * 80)
        report_lines.append(f"   Verbatims enrichis: {len(llm_results['llm_df'])}")
        if llm_results.get('cache_hit_rate'):
            report_lines.append(f"   Cache hit rate: {llm_results['cache_hit_rate']:.1f}%")
        if llm_results.get('parse_rate'):
            report_lines.append(f"   Parse success rate: {llm_results['parse_rate']:.1f}%")
        report_lines.append("")
        if llm_results.get('theme_distribution') is not None:
            report_lines.append("   Distribution des thèmes:")
            for theme, count in llm_results['theme_distribution'].head(8).items():
                pct = (count / len(llm_results['llm_df'])) * 100
                report_lines.append(f"     * {theme}: {count} ({pct:.1f}%)")
            report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("FIN DU RAPPORT")
    report_lines.append("=" * 80)
    
    # Sauvegarde du rapport
    report_text = "\n".join(report_lines)
    with open(f'{output_dir}/rapport_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[OK] Rapport de synthèse sauvegardé: {output_dir}/rapport_synthese.txt\n")
    print(report_text)


# ===================================================================================
# FONCTION PRINCIPALE
# ===================================================================================

def main():
    """
    Fonction principale qui orchestre toute l'analyse CSAT.
    """
    print("\n" + "=" * 80)
    print("  ANALYSE CSAT - SCRIPT COMPLET POUR ENTRETIEN TECHNIQUE")
    print("=" * 80)
    print("\nCe script effectue une analyse complète des données CSAT en 6 étapes:")
    print("  1. Chargement et exploration des données")
    print("  2. Analyse NLP classique (BoW, TF-IDF, n-grams)")
    print("  3. Modèle baseline supervisé (classification)")
    print("  4. Analyse thématique (LDA)")
    print("  5. Enrichissement LLM (optionnel - si clé API disponible)")
    print("  6. Génération du rapport de synthèse")
    print("\n" + "=" * 80 + "\n")
    
    # Création du dossier de sortie
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dossier de sortie: {output_dir}\n")
    
    # ÉTAPE 1: Chargement et exploration
    print("=" * 80)
    print("ÉTAPE 1/6 : CHARGEMENT ET EXPLORATION DES DONNÉES")
    print("=" * 80 + "\n")
    
    df = load_data_from_sqlite(CONFIG['db_path'], CONFIG['table_name'])
    data_quality = analyze_data_quality(df)
    csat_by_offer = explore_distributions(df, output_dir)
    
    # ÉTAPE 2: Extraction des verbatims négatifs et analyse NLP
    print("=" * 80)
    print("ÉTAPE 2/6 : ANALYSE NLP CLASSIQUE")
    print("=" * 80 + "\n")
    
    neg_df = extract_negative_reviews(
        df, 
        csat_threshold=CONFIG['csat_threshold'],
        nps_threshold=CONFIG['nps_threshold']
    )
    
    nlp_results = analyze_classical_nlp(
        neg_df,
        top_n=CONFIG['top_n_terms'],
        output_dir=output_dir
    )
    
    # ÉTAPE 3: Modèle baseline
    print("=" * 80)
    print("ÉTAPE 3/6 : MODÈLE BASELINE SUPERVISÉ")
    print("=" * 80 + "\n")
    
    baseline_results = train_baseline_classifier(
        df,
        csat_threshold=CONFIG['csat_threshold'],
        nps_threshold=CONFIG['nps_threshold'],
        output_dir=output_dir
    )
    
    # ÉTAPE 4: Analyse LDA
    print("=" * 80)
    print("ÉTAPE 4/6 : ANALYSE THÉMATIQUE (LDA)")
    print("=" * 80 + "\n")
    
    lda_results = run_lda_topic_modeling(
        neg_df,
        n_topics=CONFIG['lda_n_topics'],
        n_top_words=CONFIG['lda_n_top_words'],
        min_df=CONFIG['lda_min_df'],
        max_df=CONFIG['lda_max_df'],
        output_dir=output_dir
    )
    
    # ÉTAPE 5: Enrichissement LLM (optionnel)
    print("=" * 80)
    print("ÉTAPE 5/6 : ENRICHISSEMENT LLM (OPTIONNEL)")
    print("=" * 80 + "\n")
    
    llm_results = run_llm_enrichment_optional(
        neg_df,
        output_dir=output_dir
    )
    
    # ÉTAPE 6: Rapport de synthèse
    print("=" * 80)
    print("ÉTAPE 6/6 : GÉNÉRATION DU RAPPORT DE SYNTHÈSE")
    print("=" * 80 + "\n")
    
    generate_summary_report(
        data_quality,
        csat_by_offer,
        nlp_results,
        baseline_results,
        lda_results,
        output_dir,
        llm_results
    )
    
    # Récapitulatif final
    print("\n" + "=" * 80)
    print("[OK] ANALYSE TERMINÉE AVEC SUCCÈS")
    print("=" * 80)
    print(f"\n[DATA] Tous les résultats sont disponibles dans le dossier: {output_dir}/")
    print("\n[GRAPH] Fichiers générés:")
    print("   Explorations:")
    print("     - csat_par_offre.csv")
    print("     - distribution_csat_globale.html")
    print("     - csat_moyen_par_offre.html")
    print("\n   Analyse NLP:")
    print("     - top_termes.csv / .html")
    print("     - top_bigrams.csv / .html")
    print("     - top_trigrams.csv")
    print("     - tfidf_scores.csv")
    print("\n   Modèle Baseline:")
    print("     - baseline_metrics.csv")
    print("     - baseline_classification_report.csv")
    print("     - baseline_top_negative_terms.csv")
    print("     - baseline_top_positive_terms.csv")
    print("\n   Analyse LDA:")
    print("     - lda_topics_summary.csv")
    print("     - lda_topics_detailed.csv")
    print("     - lda_topic_distribution.csv")
    print("     - verbatims_negatifs_avec_topics.csv")
    print("     - lda_topics_visualization.html")
    
    # Vérifier si les fichiers LLM ont été générés
    if os.path.exists(f'{output_dir}/llm_enrichment_results.csv'):
        print("\n   Enrichissement LLM:")
        print("     - llm_enrichment_results.jsonl")
        print("     - llm_enrichment_results.csv")
        print("     - Cache: outputs/llm_cache.jsonl")
    
    print("\n   Synthèse:")
    print("     - rapport_synthese.txt")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
