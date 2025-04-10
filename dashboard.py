import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Ancrage Territorial ",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
@st.cache_data
def load_data():
    """Charger les données"""
    df = pd.read_csv('merged_data.csv')
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y')
    df['Year'] = df['TIME_PERIOD'].dt.year
    return df

@st.cache_resource
def load_model():
    """Charger le modèle"""
    with open('model_xgboost.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('feature_info_ancrage_territorial.pkl', 'rb') as file:
        feature_info = pickle.load(file)
    
    return model, feature_info

# Charger les données et le modèle
try:
    df = load_data()
    model, feature_info = load_model()
    data_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement des données ou du modèle: {e}")
    data_loaded = False

# CSS personnalisé pour le thème Perpignan
st.markdown("""
<style>
    /* Couleurs thématiques de Perpignan */
    :root {
        --primary-color: #D32F2F;         /* Rouge de Perpignan */
        --secondary-color: #FFC107;       /* Jaune catalan */
        --background-color: #F5F5F5;      /* Fond clair */
        --text-color: #333333;            /* Texte foncé */
        --accent-color: #1976D2;          /* Bleu méditerranéen */
    }
    
    /* Fond principal */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* En-têtes */
    h1, h2, h3 {
        color: var(--primary-color);
        font-family: 'Georgia', serif;
        text-align: center;  /* Centrer tous les titres */
    }
    
    /* Sous-titres */
    h4, h5, h6 {
        color: var(--text-color);
        font-family: 'Arial', sans-serif;
    }
    
    /* Cartes pour les métriques */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        height: 100%;  /* Assurer une hauteur uniforme */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center; /* Centrer horizontalement le contenu */
        text-align: center;
    }
    
    /* Style pour les valeurs dans les cartes métriques */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-top: 10px;
        display: block; /* Assure que la valeur est bien dans le flux */
    }
    
    /* Style pour les titres de métriques */
    .metric-title {
        font-size: 1rem;
        font-weight: bold;
        display: block; /* Assure que le titre est bien dans le flux */
    }
    
    /* Barre latérale */
    .css-1d391kg {
        background-color: #FFFFFF;
    }
    
    /* Centrer le titre principal */
    .main-title {
        text-align: center;
        width: 100%;
    }
    
    /* Centrer le sous-titre */
    .sub-title {
        text-align: center;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Logo et titre
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("logo.png", width=200)  
with col2:
    st.markdown("<h1 class='main-title'>Ancrage Territorial de Perpignan</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>Analyse et étude du territoire</h3>", unsafe_allow_html=True)

st.markdown("---")

# Barre latérale
st.sidebar.image("logo.png", width=150)
st.sidebar.title("Navigation")

# Menu de navigation
page = st.sidebar.radio(
    "",
    ["Tableau de bord", "Prédiction", "Analyse des facteurs"]
)

# Séparateur
st.sidebar.markdown("---")

# Filtres
if data_loaded:
    st.sidebar.subheader("Filtres")
    
    # Années
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.multiselect("Sélectionner les années", years, default=years[-3:])
    
    # Sexe
    sexes = df['Sexe'].unique()
    selected_sexes = st.sidebar.multiselect("Sélectionner par sexe", sexes, default=sexes)
    
    # Tranche d'âge si disponible
    if 'age' in df.columns:
        age_groups = df['age'].unique()
        selected_age_groups = st.sidebar.multiselect("Sélectionner les tranches d'âge", age_groups, default=age_groups)
    else:
        selected_age_groups = None
    
    # Filtrer les données
    filtered_df = df.copy()
    if selected_years and len(selected_years) > 0:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
    if selected_sexes and len(selected_sexes) > 0:
        filtered_df = filtered_df[filtered_df['Sexe'].isin(selected_sexes)]
    if selected_age_groups is not None and len(selected_age_groups) > 0:
        filtered_df = filtered_df[filtered_df['age'].isin(selected_age_groups)]

# Pages
if page == "Tableau de bord":
    if data_loaded and not filtered_df.empty:
        st.markdown("<h2>Tableau de bord de Perpignan</h2>", unsafe_allow_html=True)
        
        # Métriques clés en haut
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'Taux_Chomage (%)' in filtered_df.columns:
                avg_unemployment = filtered_df['Taux_Chomage (%)'].mean()
                if not pd.isna(avg_unemployment):
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Taux de chômage moyen</div>
                        <div class='metric-value'>{avg_unemployment:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Taux de chômage moyen</div>
                        <div class='metric-value'>N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-title'>Taux de chômage moyen</div>
                    <div class='metric-value'>N/A</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if 'Population_Active' in filtered_df.columns:
                total_active_pop = filtered_df['Population_Active'].sum()
                if not pd.isna(total_active_pop):
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Population active totale</div>
                        <div class='metric-value'>{total_active_pop:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Population active totale</div>
                        <div class='metric-value'>N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-title'>Population active totale</div>
                    <div class='metric-value'>N/A</div>
                </div>
                """, unsafe_allow_html=True)
                    
        with col3:
            if 'company_count' in filtered_df.columns:
                avg_companies = filtered_df['company_count'].mean()
                if not pd.isna(avg_companies):
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Nombre moyen d'entreprises</div>
                        <div class='metric-value'>{avg_companies:.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Nombre moyen d'entreprises</div>
                        <div class='metric-value'>N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-title'>Nombre moyen d'entreprises</div>
                    <div class='metric-value'>N/A</div>
                </div>
                """, unsafe_allow_html=True)
                    
        with col4:
            if 'avg_CRE_Measure' in filtered_df.columns:
                avg_cre = filtered_df['avg_CRE_Measure'].mean()
                if not pd.isna(avg_cre):
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Mesure CRE moyenne</div>
                        <div class='metric-value'>{avg_cre:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Mesure CRE moyenne</div>
                        <div class='metric-value'>N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-title'>Mesure CRE moyenne</div>
                    <div class='metric-value'>N/A</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Évolution du taux de chômage")
            # Agréger par année et sexe
            if 'Taux_Chomage (%)' in filtered_df.columns and 'Year' in filtered_df.columns and 'Sexe' in filtered_df.columns:
                unemployment_trend = filtered_df.groupby(['Year', 'Sexe'])['Taux_Chomage (%)'].mean().reset_index()
                
                fig = px.line(unemployment_trend, x='Year', y='Taux_Chomage (%)', 
                             color='Sexe', markers=True,
                             title="Évolution du taux de chômage par sexe à Perpignan")
                fig.update_layout(xaxis_title="Année", yaxis_title="Taux de chômage (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour afficher l'évolution du taux de chômage")
        
        with col2:
            st.subheader("Population active vs. Emploi")
            # Agréger par année
            if 'Population_Active' in filtered_df.columns and 'Employed_Total' in filtered_df.columns and 'Year' in filtered_df.columns:
                employment_data = filtered_df.groupby('Year')[['Population_Active', 'Employed_Total']].mean().reset_index()
                
                fig = px.bar(employment_data, x='Year', 
                             y=['Population_Active', 'Employed_Total'],
                             barmode='group',
                             title="Population active et emploi par année")
                fig.update_layout(xaxis_title="Année", yaxis_title="Nombre de personnes")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour afficher la comparaison population active vs. emploi")
        
        # Carte choroplèthe ou carte de densité - Remplacer par des données réelles
        st.subheader("Distribution géographique des entreprises à Perpignan")
        st.markdown("*Cette carte montre une simulation de la distribution des entreprises dans les différents quartiers de Perpignan.*")
        
        # Exemple de carte simple avec Plotly - À adapter avec vos données géospatiales réelles
        quartiers = ["Centre-Ville", "Bastide Saint-Jacques", "Les Remparts", "Haut-Vernet","Bas-Vernet", 
                            "Saint-Charles","Moulin à Vent", "Saint-Martin", "La Bastide","Catalunya"]
        
        entreprises = np.random.randint(50, 500, size=len(quartiers))
        
        fig = px.bar(x=quartiers, y=entreprises, 
                    title="Nombre d'entreprises par quartier", 
                    labels={'x': 'Quartier', 'y': 'Nombre d\'entreprises'})
        fig.update_traces(marker_color='#D32F2F')
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparaison des indicateurs économiques
        st.subheader("Tableau comparatif des indicateurs économiques")
        if 'Taux_Chomage (%)' in filtered_df.columns and 'avg_CRE_Measure' in filtered_df.columns:
            economic_data = filtered_df.groupby(['Year', 'Sexe'])[['Taux_Chomage (%)', 'avg_CRE_Measure']].mean().reset_index()
            economic_data = economic_data.sort_values(by=['Year', 'Sexe'])
            st.dataframe(economic_data, use_container_width=True)
        else:
            st.warning("Données insuffisantes pour afficher le tableau comparatif")
    
    else:
        st.warning("Données non disponibles ou aucune donnée ne correspond aux filtres sélectionnés. Veuillez vérifier le chargement des fichiers ou modifier vos filtres.")

elif page == "Prédiction":
    st.markdown("<h2>Prédiction du taux de chômage</h2>", unsafe_allow_html=True)
    
    if data_loaded:
        st.info("Utilisez ce formulaire pour prédire le taux de chômage en fonction des paramètres socio-économiques.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Année", min_value=2017, max_value=2030, value=2025)
            employed_total = st.number_input("Nombre total d'employés", min_value=10000, max_value=100000, value=55000)
            population_active = st.number_input("Population active", min_value=10000, max_value=100000, value=60000)
        
        with col2:
            company_count = st.number_input("Nombre d'entreprises", min_value=500, max_value=2000, value=1200)
            avg_cre_measure = st.number_input("Mesure CRE moyenne", min_value=40.0, max_value=60.0, value=50.0)
            sexe = st.selectbox("Sexe", ["F", "M"])
            sexe_m = 1 if sexe == "M" else 0
        
        if st.button("Prédire le taux de chômage", type="primary"):
            try:
                # Préparer les données pour la prédiction
                input_data = pd.DataFrame({
                    'Year': [year],
                    'Employed_Total': [employed_total],
                    'Population_Active': [population_active],
                    'company_count': [company_count],
                    'avg_CRE_Measure': [avg_cre_measure],
                    'Sexe_M': [sexe_m]
                })
                
                # Faire la prédiction
                prediction = model.predict(input_data)[0]
                
                # Afficher le résultat avec une jauge
                st.subheader("Résultat de la prédiction")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Taux de chômage prédit (%)"},
                        gauge = {
                            'axis': {'range': [0, 15]},
                            'bar': {'color': "#D32F2F"},
                            'steps': [
                                {'range': [0, 5], 'color': "lightgreen"},
                                {'range': [5, 10], 'color': "lightyellow"},
                                {'range': [10, 15], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Création d'une seule carte en HTML pour afficher prédiction et interprétation
                    if prediction < 5:
                        interpretation = "Taux de chômage faible, situation économique favorable."
                    elif prediction < 8:
                        interpretation = "Taux de chômage modéré, situation économique stable."
                    elif prediction < 12:
                        interpretation = "Taux de chômage élevé, situation économique préoccupante."
                    else:
                        interpretation = "Taux de chômage très élevé, situation économique critique."
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Taux de chômage prédit</div>
                        <div class='metric-value'>{prediction:.2f}%</div>
                        <p style="margin-top: 10px;"><strong>Interprétation:</strong> {interpretation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparaison avec les moyennes historiques
                st.subheader("Comparaison avec les moyennes historiques")
                
                if 'Taux_Chomage (%)' in df.columns:
                    hist_avg = df['Taux_Chomage (%)'].mean()
                    hist_med = df['Taux_Chomage (%)'].median()
                    hist_min = df['Taux_Chomage (%)'].min()
                    hist_max = df['Taux_Chomage (%)'].max()
                    
                    comparison_data = {
                        'Métrique': ['Prédiction', 'Moyenne historique', 'Médiane historique', 'Minimum historique', 'Maximum historique'],
                        'Taux de chômage (%)': [prediction, hist_avg, hist_med, hist_min, hist_max]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig = px.bar(comparison_df, x='Métrique', y='Taux de chômage (%)', 
                               title="Comparaison avec les statistiques historiques",
                               color='Métrique', 
                               color_discrete_map={
                                   'Prédiction': '#D32F2F',
                                   'Moyenne historique': '#1976D2',
                                   'Médiane historique': '#388E3C',
                                   'Minimum historique': '#7B1FA2',
                                   'Maximum historique': '#F57C00'
                               })
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Données historiques insuffisantes pour la comparaison")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
    else:
        st.warning("Modèle de prédiction non disponible. Veuillez vérifier le chargement du modèle.")

elif page == "Analyse des facteurs":
    st.markdown("<h2>Analyse des facteurs d'influence</h2>", unsafe_allow_html=True)
    
    if data_loaded:
        # Importance des caractéristiques
        st.subheader("Importance des facteurs dans la prédiction du taux de chômage")
        
        # Créer un DataFrame pour l'importance des caractéristiques
        try:
            importance_df = pd.DataFrame({
                'Feature': feature_info['column_names'],
                'Importance': [feature_info['feature_importance'][col] for col in feature_info['column_names']]
            }).sort_values(by='Importance', ascending=False)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Graphique d'importance des caractéristiques
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title="Importance des facteurs dans le modèle XGBoost",
                           color='Importance', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
    <div class='metric-card'>
        <h4 class='metric-title'>Interprétation des facteurs</h4>
        <p>Les facteurs les plus importants dans la prédiction du taux de chômage à Perpignan sont :</p>
        <ol>
            <li><strong>Population Active</strong> : La taille de la population active influence directement les dynamiques du marché du travail.</li>
            <li><strong>Nombre d'employés</strong> : Le nombre total de personnes employées est un indicateur direct de la santé économique.</li>
            <li><strong>Mesure CRE</strong> : Cet indicateur composite reflète la résilience économique du territoire.</li>
            <li><strong>Nombre d'entreprises</strong> : Un indicateur de la vitalité du tissu économique local.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Erreur lors de l'affichage de l'importance des caractéristiques: {e}")
        
        # Corrélation entre les variables
        st.subheader("Corrélation entre les variables")
        
        # Calculer la matrice de corrélation
        try:
            correlation_cols = [col for col in ['Taux_Chomage (%)', 'Employed_Total', 'Population_Active', 'company_count', 'avg_CRE_Measure'] if col in filtered_df.columns]
            
            if len(correlation_cols) > 1:
                corr_matrix = filtered_df[correlation_cols].corr()
                
                # Afficher la heatmap
                fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                              title="Matrice de corrélation entre les variables économiques")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour calculer les corrélations")
        
        except Exception as e:
            st.error(f"Erreur lors du calcul de la matrice de corrélation: {e}")
        
        # Analyse bivariée
        st.subheader("Relation entre les variables clés et le taux de chômage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Employés vs Taux de chômage
            if 'Employed_Total' in filtered_df.columns and 'Taux_Chomage (%)' in filtered_df.columns and 'Sexe' in filtered_df.columns:
                fig = px.scatter(filtered_df, x='Employed_Total', y='Taux_Chomage (%)', 
                               color='Sexe', trendline="ols",
                               title="Relation entre le nombre d'employés et le taux de chômage")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour l'analyse bivariée des employés")
        
        with col2:
            # Nombre d'entreprises vs Taux de chômage
            if 'company_count' in filtered_df.columns and 'Taux_Chomage (%)' in filtered_df.columns and 'Sexe' in filtered_df.columns:
                fig = px.scatter(filtered_df, x='company_count', y='Taux_Chomage (%)', 
                               color='Sexe', trendline="ols",
                               title="Relation entre le nombre d'entreprises et le taux de chômage")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour l'analyse bivariée des entreprises")
    
    else:
        st.warning("Données non disponibles. Veuillez vérifier le chargement des fichiers.")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666;">
    © 2025 | Projet de Fin d'Études - Ancrage Territorial de Perpignan | Tous droits réservés
</div>
""", unsafe_allow_html=True)