"""
===================================================================================
DASHBOARD STREAMLIT - ANALYSE DES RÉSULTATS LLM
===================================================================================

Dashboard interactif pour explorer les résultats de l'enrichissement LLM
sur les verbatims CSAT négatifs.

FONCTIONNALITÉS :
-----------------
1. Vue d'ensemble : Statistiques globales et KPIs
2. Distribution des thèmes : Graphiques interactifs
3. Analyse de sévérité : Distribution et corrélations
4. Explorer les verbatims : Filtrage par thème/sévérité
5. Recommandations : Actions prioritaires
6. Comparaison LDA vs LLM : Croisement des analyses

UTILISATION :
-------------
streamlit run dashboard_llm_streamlit.py

Le dashboard charge automatiquement les fichiers depuis output_script/

Auteur: Analyse CSAT
Date: Mars 2026
===================================================================================
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ===================================================================================
# CONFIGURATION
# ===================================================================================

st.set_page_config(
    page_title="Dashboard LLM - Analyse CSAT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins des fichiers
OUTPUT_DIR = "output_script"
LLM_FILE = f"{OUTPUT_DIR}/llm_enrichment_results.csv"
LDA_FILE = f"{OUTPUT_DIR}/verbatims_negatifs_avec_topics.csv"


# ===================================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ===================================================================================

@st.cache_data
def load_llm_data() -> Optional[pd.DataFrame]:
    """
    Charge les résultats de l'enrichissement LLM.
    
    Returns:
        DataFrame avec les verbatims enrichis ou None si fichier absent
    """
    if not os.path.exists(LLM_FILE):
        return None
    
    try:
        df = pd.read_csv(LLM_FILE, encoding='utf-8-sig')
        
        # Nettoyage et conversion des types
        if 'severity' in df.columns:
            df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
        
        if 'confidence' in df.columns:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        
        # Filtrer les lignes avec thème valide (non vide)
        if 'theme' in df.columns:
            df = df[df['theme'].notna() & (df['theme'] != '')]
        
        return df
    
    except Exception as e:
        st.error(f"Erreur lors du chargement de {LLM_FILE}: {str(e)}")
        return None


@st.cache_data
def load_lda_data() -> Optional[pd.DataFrame]:
    """
    Charge les résultats de l'analyse LDA pour comparaison.
    
    Returns:
        DataFrame avec les verbatims et leurs topics LDA ou None si fichier absent
    """
    if not os.path.exists(LDA_FILE):
        return None
    
    try:
        df = pd.read_csv(LDA_FILE, encoding='utf-8-sig')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement de {LDA_FILE}: {str(e)}")
        return None


# ===================================================================================
# FONCTIONS D'ANALYSE ET VISUALISATION
# ===================================================================================

def display_kpi_metrics(df: pd.DataFrame):
    """
    Affiche les KPIs principaux en haut du dashboard.
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📝 Verbatims enrichis",
            value=len(df)
        )
    
    with col2:
        if 'theme' in df.columns:
            n_themes = df['theme'].nunique()
            st.metric(
                label="🏷️ Thèmes identifiés",
                value=n_themes
            )
    
    with col3:
        if 'severity' in df.columns:
            avg_severity = df['severity'].mean()
            st.metric(
                label="⚠️ Sévérité moyenne",
                value=f"{avg_severity:.2f} / 5"
            )
    
    with col4:
        if 'confidence' in df.columns:
            avg_confidence = df['confidence'].mean()
            st.metric(
                label="✅ Confiance moyenne",
                value=f"{avg_confidence:.2%}"
            )
    
    with col5:
        if 'parse_ok' in df.columns:
            success_rate = df['parse_ok'].mean()
            st.metric(
                label="🎯 Taux de succès",
                value=f"{success_rate:.1%}"
            )


def plot_theme_distribution(df: pd.DataFrame):
    """
    Affiche la distribution des thèmes identifiés.
    """
    if 'theme' not in df.columns:
        st.warning("Colonne 'theme' non trouvée dans les données")
        return
    
    theme_counts = df['theme'].value_counts().reset_index()
    theme_counts.columns = ['theme', 'count']
    theme_counts['percentage'] = (theme_counts['count'] / len(df) * 100).round(1)
    
    fig = px.bar(
        theme_counts,
        x='count',
        y='theme',
        orientation='h',
        title="Distribution des thèmes identifiés par le LLM",
        labels={'count': 'Nombre de verbatims', 'theme': 'Thème'},
        text='percentage',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    return theme_counts


def plot_severity_distribution(df: pd.DataFrame):
    """
    Affiche la distribution de la sévérité.
    """
    if 'severity' not in df.columns:
        st.warning("Colonne 'severity' non trouvée dans les données")
        return
    
    # Histogramme de sévérité
    severity_counts = df['severity'].value_counts().sort_index().reset_index()
    severity_counts.columns = ['severity', 'count']
    
    fig = px.bar(
        severity_counts,
        x='severity',
        y='count',
        title="Distribution de la sévérité (échelle 1-5)",
        labels={'severity': 'Niveau de sévérité', 'count': 'Nombre de verbatims'},
        color='severity',
        color_continuous_scale='RdYlGn_r',
        text='count'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_severity_by_theme(df: pd.DataFrame):
    """
    Affiche la sévérité moyenne par thème.
    """
    if 'severity' not in df.columns or 'theme' not in df.columns:
        return
    
    severity_by_theme = df.groupby('theme').agg({
        'severity': ['mean', 'median', 'count']
    }).reset_index()
    
    severity_by_theme.columns = ['theme', 'severity_mean', 'severity_median', 'count']
    severity_by_theme = severity_by_theme.sort_values('severity_mean', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=severity_by_theme['severity_mean'],
        y=severity_by_theme['theme'],
        orientation='h',
        name='Sévérité moyenne',
        marker=dict(color=severity_by_theme['severity_mean'], colorscale='RdYlGn_r'),
        text=severity_by_theme['severity_mean'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Sévérité moyenne: %{x:.2f}<br>Nombre: %{customdata}<extra></extra>',
        customdata=severity_by_theme['count']
    ))
    
    fig.update_layout(
        title="Sévérité moyenne par thème",
        xaxis_title="Sévérité moyenne",
        yaxis_title="Thème",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return severity_by_theme


def plot_confidence_analysis(df: pd.DataFrame):
    """
    Analyse de la confiance des prédictions LLM.
    """
    if 'confidence' not in df.columns:
        st.warning("Colonne 'confidence' non trouvée dans les données")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution de la confiance
        fig1 = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title="Distribution de la confiance du LLM",
            labels={'confidence': 'Score de confiance', 'count': 'Fréquence'},
            color_discrete_sequence=['#636EFA']
        )
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Boxplot confiance par thème
        if 'theme' in df.columns:
            fig2 = px.box(
                df,
                x='theme',
                y='confidence',
                title="Confiance par thème",
                labels={'confidence': 'Score de confiance', 'theme': 'Thème'},
                color='theme'
            )
            fig2.update_layout(height=350, showlegend=False)
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)


def plot_heatmap_theme_severity(df: pd.DataFrame):
    """
    Heatmap croisant thèmes et niveaux de sévérité.
    """
    if 'theme' not in df.columns or 'severity' not in df.columns:
        return
    
    # Créer la matrice de contingence
    contingency = pd.crosstab(df['theme'], df['severity'])
    
    # Calculer les pourcentages par thème
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=contingency_pct.values,
        x=[f"Sévérité {int(col)}" for col in contingency_pct.columns],
        y=contingency_pct.index,
        colorscale='RdYlGn_r',
        text=contingency_pct.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Pourcentage"),
        hovertemplate='<b>%{y}</b><br>%{x}<br>%{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Distribution de la sévérité par thème (%)",
        xaxis_title="Niveau de sévérité",
        yaxis_title="Thème",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_recommendations_summary(df: pd.DataFrame):
    """
    Affiche un résumé des recommandations par thème.
    """
    if 'theme' not in df.columns or 'recommended_action' not in df.columns:
        st.warning("Colonnes nécessaires non trouvées")
        return
    
    st.subheader("💡 Top recommandations par thème")
    
    # Grouper par thème et extraire les recommandations les plus fréquentes
    theme_reco = df.groupby('theme')['recommended_action'].apply(
        lambda x: x.value_counts().head(3)
    ).reset_index()
    
    theme_reco.columns = ['theme', 'recommended_action', 'frequency']
    
    # Afficher par thème
    themes = theme_reco['theme'].unique()
    
    for theme in themes:
        with st.expander(f"🏷️ {theme}"):
            theme_data = theme_reco[theme_reco['theme'] == theme]
            
            if len(theme_data) > 0:
                for idx, row in theme_data.iterrows():
                    st.markdown(f"- **{row['recommended_action']}** ({row['frequency']} occurrences)")
            else:
                st.info("Aucune recommandation disponible")


def display_verbatim_explorer(df: pd.DataFrame):
    """
    Interface pour explorer les verbatims enrichis avec filtres.
    """
    st.subheader("🔍 Explorer les verbatims enrichis")
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'theme' in df.columns:
            themes = ['Tous'] + sorted(df['theme'].unique().tolist())
            selected_theme = st.selectbox("Filtrer par thème", themes)
    
    with col2:
        if 'severity' in df.columns:
            severity_range = st.slider(
                "Filtrer par sévérité",
                min_value=int(df['severity'].min()),
                max_value=int(df['severity'].max()),
                value=(int(df['severity'].min()), int(df['severity'].max()))
            )
    
    with col3:
        if 'confidence' in df.columns:
            confidence_threshold = st.slider(
                "Confiance minimale",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
    
    # Appliquer les filtres
    filtered_df = df.copy()
    
    if 'theme' in df.columns and selected_theme != 'Tous':
        filtered_df = filtered_df[filtered_df['theme'] == selected_theme]
    
    if 'severity' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['severity'] >= severity_range[0]) &
            (filtered_df['severity'] <= severity_range[1])
        ]
    
    if 'confidence' in df.columns:
        filtered_df = filtered_df[filtered_df['confidence'] >= confidence_threshold]
    
    st.info(f"📊 {len(filtered_df)} verbatims correspondent aux critères")
    
    # Tri
    sort_by = st.selectbox(
        "Trier par",
        ['severity', 'confidence', 'theme'] if 'severity' in df.columns else ['theme']
    )
    
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    # Affichage des verbatims
    if len(filtered_df) > 0:
        # Nombre de verbatims à afficher
        n_display = st.slider("Nombre de verbatims à afficher", 5, 50, 10)
        
        for idx, row in filtered_df.head(n_display).iterrows():
            with st.expander(
                f"🏷️ {row.get('theme', 'N/A')} | "
                f"⚠️ Sévérité: {row.get('severity', 'N/A')}/5 | "
                f"✅ Confiance: {row.get('confidence', 0):.1%}"
            ):
                st.markdown(f"**Verbatim :**")
                st.write(row.get('commentaire', 'N/A'))
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"**Sous-thème :** {row.get('sub_theme', 'N/A')}")
                    st.markdown(f"**Problème principal :** {row.get('main_issue', 'N/A')}")
                
                with col_b:
                    st.markdown(f"**Action recommandée :** {row.get('recommended_action', 'N/A')}")
                    
                    # Métadonnées de traçabilité si disponibles
                    if 'contact_id' in row and pd.notna(row['contact_id']):
                        st.markdown(f"**ID Contact :** {row['contact_id']}")
    else:
        st.warning("Aucun verbatim ne correspond aux critères de filtrage")


def compare_lda_llm(llm_df: pd.DataFrame, lda_df: pd.DataFrame):
    """
    Compare les résultats de l'analyse LDA avec les thèmes LLM.
    """
    st.subheader("🔄 Comparaison LDA vs LLM")
    
    if llm_df is None or lda_df is None:
        st.warning("Données LDA ou LLM manquantes pour la comparaison")
        return
    
    st.info(
        "Cette section compare les topics découverts par LDA (non supervisé) "
        "avec les thèmes identifiés par le LLM (supervisé avec taxonomie fixe)."
    )
    
    # Vérifier si on peut joindre les DataFrames
    # On suppose que l'index ou une colonne commune existe
    
    # Pour l'instant, afficher les statistiques séparées
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Analyse LDA**")
        if 'dominant_topic' in lda_df.columns:
            n_topics = lda_df['dominant_topic'].nunique()
            st.metric("Nombre de topics LDA", n_topics)
            
            topic_dist = lda_df['dominant_topic'].value_counts().sort_index()
            fig = px.bar(
                x=topic_dist.index,
                y=topic_dist.values,
                labels={'x': 'Topic LDA', 'y': 'Nombre de documents'},
                title="Distribution des topics LDA"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**🤖 Analyse LLM**")
        if 'theme' in llm_df.columns:
            n_themes = llm_df['theme'].nunique()
            st.metric("Nombre de thèmes LLM", n_themes)
            
            theme_dist = llm_df['theme'].value_counts()
            fig = px.bar(
                x=theme_dist.values,
                y=theme_dist.index,
                orientation='h',
                labels={'x': 'Nombre de verbatims', 'y': 'Thème'},
                title="Distribution des thèmes LLM"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================================
# INTERFACE PRINCIPALE
# ===================================================================================

def main():
    """
    Fonction principale du dashboard Streamlit.
    """
    
    # En-tête
    st.title("📊 Dashboard d'Analyse LLM - Verbatims CSAT")
    st.markdown("---")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        llm_df = load_llm_data()
        lda_df = load_lda_data()
    
    # Vérification des données
    if llm_df is None:
        st.error(
            f"❌ Fichier {LLM_FILE} introuvable ou vide.\n\n"
            "Assurez-vous d'avoir exécuté le script d'analyse complet "
            "et que l'enrichissement LLM a réussi."
        )
        
        st.info(
            "**Pour générer les données LLM :**\n"
            "1. Vérifiez votre clé API OpenAI dans `.env`\n"
            "2. Assurez-vous d'avoir du crédit sur votre compte OpenAI\n"
            "3. Supprimez le cache corrompu si nécessaire : `outputs/llm_cache.jsonl`\n"
            "4. Relancez le script : `python analyse_csat_complete_standalone.py`"
        )
        return
    
    if len(llm_df) == 0:
        st.warning(
            "⚠️ Le fichier LLM existe mais ne contient aucun verbatim enrichi.\n\n"
            "Cela peut signifier que tous les appels API ont échoué (quota dépassé, erreur réseau, etc.)"
        )
        return
    
    st.success(f"✅ {len(llm_df)} verbatims enrichis chargés avec succès")
    
    # Sidebar pour navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choisir une section",
        [
            "📈 Vue d'ensemble",
            "🏷️ Distribution des thèmes",
            "⚠️ Analyse de sévérité",
            "✅ Analyse de confiance",
            "🔍 Explorer les verbatims",
            "💡 Recommandations",
            "🔄 Comparaison LDA vs LLM"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informations")
    st.sidebar.info(
        f"**Fichiers source :**\n"
        f"- LLM : `{LLM_FILE}`\n"
        f"- LDA : `{LDA_FILE if lda_df is not None else 'Non disponible'}`"
    )
    
    # Affichage selon la page sélectionnée
    if page == "📈 Vue d'ensemble":
        st.header("📈 Vue d'ensemble")
        display_kpi_metrics(llm_df)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des thèmes")
            if 'theme' in llm_df.columns:
                theme_counts = llm_df['theme'].value_counts()
                fig = px.pie(
                    values=theme_counts.values,
                    names=theme_counts.index,
                    title="Répartition des verbatims par thème"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribution de la sévérité")
            if 'severity' in llm_df.columns:
                severity_counts = llm_df['severity'].value_counts().sort_index()
                fig = px.pie(
                    values=severity_counts.values,
                    names=[f"Sévérité {int(x)}" for x in severity_counts.index],
                    title="Répartition par niveau de sévérité",
                    color_discrete_sequence=px.colors.sequential.RdBu_r
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        st.markdown("---")
        st.subheader("📊 Statistiques descriptives")
        
        stats_cols = ['severity', 'confidence']
        available_stats = [col for col in stats_cols if col in llm_df.columns]
        
        if available_stats:
            st.dataframe(llm_df[available_stats].describe(), use_container_width=True)
    
    elif page == "🏷️ Distribution des thèmes":
        st.header("🏷️ Distribution des thèmes")
        theme_counts = plot_theme_distribution(llm_df)
        
        if theme_counts is not None:
            st.markdown("---")
            st.subheader("📋 Tableau détaillé")
            st.dataframe(theme_counts, use_container_width=True)
    
    elif page == "⚠️ Analyse de sévérité":
        st.header("⚠️ Analyse de sévérité")
        
        plot_severity_distribution(llm_df)
        
        st.markdown("---")
        
        severity_by_theme = plot_severity_by_theme(llm_df)
        
        if severity_by_theme is not None:
            st.markdown("---")
            st.subheader("📋 Sévérité par thème - Tableau détaillé")
            st.dataframe(severity_by_theme, use_container_width=True)
        
        st.markdown("---")
        
        plot_heatmap_theme_severity(llm_df)
    
    elif page == "✅ Analyse de confiance":
        st.header("✅ Analyse de confiance")
        plot_confidence_analysis(llm_df)
        
        st.markdown("---")
        
        # Analyse des verbatims à faible confiance
        if 'confidence' in llm_df.columns:
            st.subheader("⚠️ Verbatims à faible confiance (< 0.7)")
            low_confidence = llm_df[llm_df['confidence'] < 0.7]
            
            if len(low_confidence) > 0:
                st.warning(f"{len(low_confidence)} verbatims ont une confiance < 0.7")
                
                # Afficher quelques exemples
                st.markdown("**Exemples de verbatims à faible confiance :**")
                for idx, row in low_confidence.head(5).iterrows():
                    with st.expander(f"Confiance: {row['confidence']:.2%} | Thème: {row.get('theme', 'N/A')}"):
                        st.write(row.get('commentaire', 'N/A'))
            else:
                st.success("Tous les verbatims ont une confiance >= 0.7")
    
    elif page == "🔍 Explorer les verbatims":
        st.header("🔍 Explorer les verbatims enrichis")
        display_verbatim_explorer(llm_df)
    
    elif page == "💡 Recommandations":
        st.header("💡 Recommandations d'actions")
        display_recommendations_summary(llm_df)
        
        # Top actions globales
        st.markdown("---")
        st.subheader("🎯 Top 10 actions recommandées (global)")
        
        if 'recommended_action' in llm_df.columns:
            top_actions = llm_df['recommended_action'].value_counts().head(10)
            
            fig = px.bar(
                x=top_actions.values,
                y=top_actions.index,
                orientation='h',
                labels={'x': 'Fréquence', 'y': 'Action recommandée'},
                title="Actions les plus fréquemment recommandées"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔄 Comparaison LDA vs LLM":
        st.header("🔄 Comparaison LDA vs LLM")
        compare_lda_llm(llm_df, lda_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Dashboard créé avec Streamlit | Analyse CSAT 2026"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
