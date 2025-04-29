import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from streamlit_option_menu import option_menu

def export_as_pdf(
    df: pd.DataFrame,
    filtered_df=None,
    model=None,
    feature_info=None,
    foundations_df: pd.DataFrame = None,
    state_subv_df: pd.DataFrame = None,
    city_subv_df: pd.DataFrame = None,
    prediction_params: dict = None,
    prediction_result: float = None
) -> bytes:
    """
    Génère un PDF complet avec :
     - Page 1 : Tableau de bord
     - Page 2 : Prédiction
     - Page 3 : Analyse des facteurs
     - Page 4 : Fondations européennes
     - Page 5 : Subventions étatiques
     - Page 6 : Aides de la Ville de Perpignan
    """
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    from datetime import datetime
    import io, matplotlib.pyplot as plt

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='SubTitle', parent=styles['Heading2'],
        fontSize=14, textColor=colors.maroon, spaceAfter=6))
    normal = styles['Normal']

    # style commun pour tous les tableaux
    common_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.maroon),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN',      (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
        ('INNERGRID',  (0, 0), (-1, -1), 0.25, colors.grey),
        ('BOX',        (0, 0), (-1, -1), 0.5, colors.black),
        ('LEFTPADDING',  (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR',  (0, 1), (-1, -1), colors.black),
    ])

    content = []
    
    # ------ PAGE 1: TABLEAU DE BORD ------
    # Titre principal
    content.append(Paragraph("Rapport d'Ancrage Territorial de Perpignan", styles['Heading1']))
    content.append(Spacer(1, 12))
    
    # Date du rapport
    content.append(Paragraph(f"Date du rapport: {datetime.now():%d/%m/%Y}", styles['Normal']))
    content.append(Spacer(1, 24))
    
    # Sous-titre pour les métriques
    content.append(Paragraph("Métriques clés", styles['SubTitle']))
    content.append(Spacer(1, 12))
    
    # Créer un tableau pour les métriques
    metrics_data = []
    headers = ["Indicateur", "Valeur"]
    metrics_data.append(headers)
    
    if filtered_df is not None:
        # Ajouter chaque métrique disponible
        if 'Taux_Chomage (%)' in filtered_df.columns:
            metrics_data.append(["Taux de chômage moyen", f"{filtered_df['Taux_Chomage (%)'].mean():.2f}%"])
        
        if 'Population_Active' in filtered_df.columns:
            metrics_data.append(["Population active totale", f"{filtered_df['Population_Active'].sum():,.0f}"])
        
        if 'company_count' in filtered_df.columns:
            metrics_data.append(["Nombre moyen d'entreprises", f"{filtered_df['company_count'].mean():.0f}"])
        
        if 'avg_CRE_Measure' in filtered_df.columns:
            metrics_data.append(["Mesure CRE moyenne", f"{filtered_df['avg_CRE_Measure'].mean():.2f}"])
    
    # Créer le tableau des métriques
    metrics_table = Table(metrics_data, colWidths=[300, 150])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.maroon),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
    ]))
    content.append(metrics_table)
    content.append(Spacer(1, 24))
    
    # Sous-titre pour les données
    content.append(Paragraph("Données du tableau de bord", styles['SubTitle']))
    content.append(Spacer(1, 12))
    
    # Tableau des données principales (10 premières lignes)
    if not df.empty:
        # Sélectionner seulement les colonnes pertinentes pour un meilleur formatage
        important_cols = ['Year', 'Sexe', 'Taux_Chomage (%)', 'Population_Active', 'Employed_Total', 'company_count']
        display_cols = [col for col in important_cols if col in df.columns]
        
        if display_cols:
            table_data = [display_cols]  # En-têtes
            
            # Ajouter jusqu'à 10 lignes de données
            for _, row in df[display_cols].head(10).iterrows():
                table_data.append([str(row[col]) for col in display_cols])
            
            data_table = Table(table_data, colWidths=[80] * len(display_cols))
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.maroon),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ]))
            content.append(data_table)
    
    # Ajouter un saut de page
    content.append(PageBreak())
    
    # ------ PAGE 2: PRÉDICTION ------
    content.append(Paragraph("Prédiction du taux de chômage", styles['Heading1']))
    content.append(Spacer(1, 12))

   # Paramètres de prédiction
    content.append(Paragraph("Paramètres de prédiction", styles['SubTitle']))
    content.append(Spacer(1, 12))

    params_data = [["Paramètre", "Valeur"]]
    if prediction_params:
        for k, v in prediction_params.items():
            params_data.append([k, str(v)])
    else:
        params_data += [
        ["Année", "2025"],
        ["Nombre total d'employés", "55000"],
        ["Population active", "60000"],
        ["Sexe", "M"],
        ["Nombre d'entreprises", "1200"],
        ["Mesure CRE moyenne", "50.0"],
        ["Tranche d'âge", "25-39"]
    ]

    params_table = Table(params_data, colWidths=[300, 150])
    params_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (1, 0), colors.maroon),
    ('TEXTCOLOR',  (0, 0), (1, 0), colors.whitesmoke),
    ('ALIGN',      (0, 0), (1, 0), 'CENTER'),
    ('FONTNAME',   (0, 0), (1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (1, 0), 12),
    ('BACKGROUND', (0, 1), (1, -1), colors.beige),
    ('GRID',       (0, 0), (-1, -1), 1, colors.black),
    ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
    ('ALIGN',      (1, 1), (1, -1), 'RIGHT'),
]))
    content.append(params_table)
    content.append(Spacer(1, 24))

    # Résultat
    content.append(Paragraph("Résultat de la prédiction", styles['SubTitle']))
    content.append(Spacer(1, 12))

    prediction_value = prediction_result if prediction_result is not None else 8.5
    
    # Image d'une jauge (simulée) - dans un cas réel, vous pourriez générer cette image dynamiquement
    # Créer une jauge simple avec matplotlib
    def create_gauge_figure(prediction_value=8.5):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1)
        ax.set_title("Taux de chômage prédit (%)")
        ax.add_patch(plt.Rectangle((0, 0.1), 5, 0.3, color='lightgreen'))
        ax.add_patch(plt.Rectangle((5, 0.1), 5, 0.3, color='lightyellow'))
        ax.add_patch(plt.Rectangle((10, 0.1), 5, 0.3, color='salmon'))
        ax.arrow(prediction_value, 0.6, 0, -0.2, head_width=0.3, head_length=0.1, fc='red', ec='red')
        ax.text(prediction_value, 0.7, f"{prediction_value:.2f}%", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Enregistrer en mémoire
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer
    
    # Créer la jauge
    gauge_img = create_gauge_figure(prediction_value)
    img_reader = ImageReader(gauge_img)
    
    # Sauvegarde et insertion de l’image
    from reportlab.platypus import Image as ReportLabImage
    temp_img_path = "temp_gauge.png"
    with open(temp_img_path, "wb") as f:
        f.write(gauge_img.getvalue())

    content.append(ReportLabImage(temp_img_path, width=400, height=200))
    content.append(Spacer(1, 12))

    # Interprétation
    if prediction_value < 5:
        interpretation = "Taux de chômage faible, situation économique favorable."
    elif prediction_value < 8:
        interpretation = "Taux de chômage modéré, situation économique stable."
    elif prediction_value < 12:
        interpretation = "Taux de chômage élevé, situation économique préoccupante."
    else:
        interpretation = "Taux de chômage très élevé, situation économique critique."

    content.append(Paragraph(f"<b>Interprétation:</b> {interpretation}", styles['Normal']))
    content.append(PageBreak())
    
    # ------ PAGE 3: ANALYSE DES FACTEURS ------
    content.append(Paragraph("Analyse des facteurs d'influence", styles['Heading1']))
    content.append(Spacer(1, 12))
    
    # Si le modèle et les informations sur les features sont disponibles
    if model is not None and feature_info is not None:
        content.append(Paragraph("Importance des facteurs dans la prédiction", styles['SubTitle']))
        content.append(Spacer(1, 12))
        
        # Créer un graphique d'importance des features avec matplotlib
        def create_feature_importance_chart(model, feature_info):
            # Extraire l'importance des features
            feature_importances = model.feature_importances_
            features = feature_info['features']
            
            # Transformer les noms pour une meilleure lisibilité
            display_names = []
            for feature in features:
                if feature.startswith('log_'):
                    display_names.append(feature[4:])  # Enlever 'log_' du début
                elif feature.startswith('age_'):
                    display_names.append(feature)  # Garder 'age_' tel quel
                else:
                    display_names.append(feature)
            
            # Créer le graphique
            fig, ax = plt.subplots(figsize=(8, 6))
            y_pos = range(len(display_names))
            ax.barh(y_pos, feature_importances, color='maroon')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_names)
            ax.set_xlabel('Importance')
            ax.set_title('Importance des facteurs dans le modèle')
            
            # Enregistrer en mémoire
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plt.close(fig)
            return img_buffer
        
        try:
            importance_img = create_feature_importance_chart(model, feature_info)
            
            # Enregistrer l'image temporairement avec une extension
            temp_importance_path = "temp_importance.png"
            with open(temp_importance_path, "wb") as f:
                f.write(importance_img.getvalue())
                
            # Ajouter l'image au document en utilisant le chemin du fichier
            content.append(ReportLabImage(temp_importance_path, width=450, height=300))
            content.append(Spacer(1, 12))
        except Exception as e:
            content.append(Paragraph(f"Erreur lors de la création du graphique d'importance: {str(e)}", styles['Normal']))
    
    # Interprétation des facteurs
    content.append(Paragraph("Interprétation des facteurs", styles['SubTitle']))
    content.append(Spacer(1, 12))
    
    interpretation_text = (
    "<b>Les facteurs les plus importants dans la prédiction du taux de chômage à Perpignan sont :</b><br/><br/>"
    "1. <b>Population Active</b> : La taille de la population active influence directement les dynamiques du marché du travail.<br/><br/>"
    "2. <b>Nombre d'employés</b> : Le nombre total de personnes employées est un indicateur direct de la santé économique.<br/><br/>"
    "3. <b>Mesure CRE</b> : Cet indicateur composite reflète la résilience économique du territoire.<br/><br/>"
    "4. <b>Nombre d'entreprises</b> : Un indicateur de la vitalité du tissu économique local."
)

    content.append(Paragraph(interpretation_text, styles['Normal']))

   # helper pour ajouter un dataframe en PDF
    def _add_dataframe_page(df_table: pd.DataFrame, title: str):
        content.append(PageBreak())
        content.append(Paragraph(title, styles['Heading1']))
        content.append(Spacer(1, 6))

    # Créer un style spécifique pour les en-têtes de colonnes (blanc sur marron)
        header_style = ParagraphStyle(
        name='HeaderStyle',
        parent=normal,
        textColor=colors.whitesmoke,  # Texte blanc
        fontName='Helvetica-Bold'
    )

    # transformer les en-têtes en Paragraph avec le style d'en-tête
        data = [[Paragraph(col, header_style) for col in df_table.columns]]
    
    # transformer les données en Paragraph avec le style normal
        for row in df_table.itertuples(index=False):
            data.append([Paragraph(str(cell), normal) for cell in row])

    # calcul dynamique de la largeur
        total_width = doc.width
        col_widths = [total_width / len(df_table.columns)] * len(df_table.columns)

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(common_style)  
        content.append(tbl)

    # --- PAGE 4: Fondations européennes ---
    if foundations_df is not None:
        _add_dataframe_page(foundations_df, "Fondations européennes")

    # --- PAGE 5: Subventions étatiques ---
    if state_subv_df is not None:
        _add_dataframe_page(state_subv_df, "Subventions étatiques")

    # --- PAGE 6: Aides et subventions Ville de Perpignan ---
    if city_subv_df is not None:
        _add_dataframe_page(city_subv_df, "Aides et subventions de la Ville de Perpignan")
    
    # Générer le document final
    doc.build(content)
    # Nettoyer les fichiers temporaires
    import os
    try:
        os.remove(temp_img_path)
        if 'temp_importance_path' in locals():
            os.remove(temp_importance_path)
    except:
        pass
        
    return buffer.getvalue()

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
    with open('xgboost_model_future.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('feature_info_future.pkl', 'rb') as file:
        feature_info = pickle.load(file)
    
    return model, feature_info

# Fonction pour charger les données des fondations
@st.cache_data
def load_foundations_data():
    """Charger les données des fondations et subventions"""
    data = {
        "Champs d'intervention": [
            "Climat environnement", 
            "Environnement climat transition énergétique",  
            "Renforcement de la cohésion économique, sociale et territoriale",
            "Principal instrument de financement pour le développement rural de l'UE",
            "Premier pilier de la politique agricole",
            "Approche de développement local intégré au FEADER"
        ],
        "Fondation ou programme": [
            "Fonds pour l'innovation",
            "Programme LIFE 2021-2027",
            "Fonds européens structurels et d'investissement (FESI)",
            "FEADER (Fonds européen agricole pour le développement rural)",
            "FEAGA (Fonds européen agricole de garantie)",
            "LEADER (Liaison entre actions de développement et l'économie rurale)"
        ],
        "Actions": [
            "Soutien des projets innovants dans les domaines du climat et de l'environnement, subventions de 2,5M à 100M€",
            "Avec un budget total de 5,4 G€, plusieurs sous-programmes : Nature, biodiversité, économie circulaire, climat…",
            "Financement de projets d'atténuation et d'adaptation au changement climatique",
            "Ces fonds renforcent la cohésion économique, sociale et territoriale au niveau régional",
            "Financement de projets locaux liés aux ODD (gestion régionale)",
            "Revitalisation des zones rurales, diversification économique et mesures agroenvironnementales"
        ]
    }
    return pd.DataFrame(data)

# Fonction pour charger les données des subventions
@st.cache_data
def load_state_subventions_data():
    """Charger les données des subventions étatiques"""
    data = {
        "Champs d'intervention": [
            # 2 x “Financements gouvernementaux”
            "Financements gouvernementaux",
            "Financements gouvernementaux",
            # 3 x “Programmes spécifiques”
            "Programmes spécifiques",
            "Programmes spécifiques",
            "Programmes spécifiques",
            # 4 x “Dotations de soutien à l'investissement”
            "Dotations de soutien à l'investissement",
            "Dotations de soutien à l'investissement",
            "Dotations de soutien à l'investissement",
            "Dotations de soutien à l'investissement",
        ],
        "Organisme": [
            "ADEME (Agence de la transition écologique)",
            "Ministère de la transition écologique",
            "MaPrimeRénov'",
            "Subventions pour la mobilité durable",
            "Aides à la protection de la biodiversité",
            "DSIL (Dotation de soutien à l'investissement)",
            "DETR (Dotation d'équipement des territoires ruraux)",  # ajout de la virgule ici
            "DPV (Dotation politique de la ville)",
            "FDVA (Fonds de développement de la vie associative)"
        ],
        "Actions": [
            "Propose plusieurs dispositifs de financement : Fonds Chaleur, Tremplin, etc.",
            "Programme LIFE : nature, climat, énergie propre…",
            "Aide financière pour la rénovation énergétique (ODD 7 & 13)",
            "Aides pour pistes cyclables, bornes électriques, transports publics",
            "Protection des écosystèmes, restauration d’habitats, espèces menacées",
            "Soutenir les projets d'investissement local",
            "Financeur des projets d'investissement dans les domaines économique, social, environnemental et touristique",
            "Subvention destinée aux communes particulièrement défavorisées et présentant des dysfonctionnements urbains.",
            "Géré au niveau national (formation) et départemental (fonctionnement et projets innovants)"
        ]
    }
    return pd.DataFrame(data)

# Fonction pour charger les données des subventions de Perpignan
@st.cache_data
def load_city_subventions_data():
    """Charger les aides et subventions de la Ville de Perpignan"""
    data = {
        "Projets": [
            "Aides “Action Cœur de Ville et Faubourg”",
            "Subventions pour la rénovation énergétique",
            "Appel à projet Unique (APU)",
            "Perp’initiatives",
            "Appel à projets pour les résidences d’artistes 2025-2026",
            "Appel à projets du Contrat de Ville Perpignan Méditerranée Métropole"
        ],
        "Programmes": [
            "Réhabilitation de l’habitat ancien dans le centre-villeet ses Faubourg (Saint-Jacques, Saint-Mathieu, La Réal, Saint-Jean, Notre Dame)",
            "Adaptation du logement pour les personnes âgées ou handicapées (jusqu’à 90 % du coût)",
            "Soutien de projets d’éducation artistique et culturelle dans la petite enfance",
            "Appel à projets citoyen lancé par la Ville de Perpignan",
            "Soutenir la création artistique",
            "Actions dans les quartiers prioritaires (appel annuel 2024-2030 via portail Dauphin)"
        ],
        "Liens":[
            "https://www.anil.org/aides-locales-travaux/details/pyrenees-orientales-aides-action-coeur-de-ville-et-faubourgs-1856/ ",
            "------------",
            "https://apu.perpignan.fr/ ",
            "https://www.mairie-perpignan.fr/fetes-et-manifestations/perpinitatives ",
            "-------------",
            "https://www.pyrenees-orientales.gouv.fr/Actions-de-l-Etat/Solidarite-hebergement-logement-et-populations-vulnerables/POLITIQUE-DE-LA-VILLE/Contrat-de-ville-PERPIGNAN-MEDITERRANEE-METROPOLE "
        ]
    }
    return pd.DataFrame(data)

# Charger les données et le modèle
try:
    df = load_data()
    model, feature_info = load_model()
    foundations_df = load_foundations_data()
    state_subv_df = load_state_subventions_data()
    city_subv_df = load_city_subventions_data()
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
    
    /* Custom styling for option menu */
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    
    /* Option menu container styling */
    div[data-testid="stSidebar"] .css-1d391kg {
        background-color: #F5F5F5;
    }
    
    /* Style for the active menu item */
    .css-1qfr0tk {
        background-color: #D32F2F !important;
    }
    
    /* Menu item text color when active */
    .css-1qfr0tk p {
        color: white !important;
    }
    
    /* Increase size of icons */
    .css-1qfr0tk svg {
        height: 1.2rem !important;
        width: 1.2rem !important;
    }
    
    /* Menu container styling */
    .css-j7qwjs {
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    /* Style pour le tableau des fondations */
    .foundation-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .foundation-table th {
        background-color: var(--primary-color);
        color: white;
        padding: 12px;
        text-align: left;
    }
    
    .foundation-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    
    .foundation-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .foundation-table tr:hover {
        background-color: #f1f1f1;
    }
    
    .foundation-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .foundation-title {
        color: var(--primary-color);
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .foundation-description {
        color: var(--text-color);
        margin-bottom: 10px;
    }
    
    .foundation-link {
        color: var(--accent-color);
        text-decoration: none;
    }
    
    .foundation-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Logo et titre
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("logo.png", width=150)  
with col2:
    st.markdown("<h1 class='main-title'>Ancrage Territorial de Perpignan</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>Analyse et étude du territoire</h3>", unsafe_allow_html=True)

st.markdown("---")

# Barre latérale
st.sidebar.image("logo.png", width=100)

# Navigation title
st.sidebar.title("Navigation")

# Using streamlit-option-menu for navigation
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # No menu title
        options=["Tableau de bord", "Prédiction", "Analyse des facteurs", "Fondations et subventions"],  # Added new option
        icons=["bar-chart-fill", "graph-up", "search", "cash-coin"],  # Added icon for foundations
        menu_icon="cast",
        default_index=0,  # Default is "Tableau de bord"
        styles={
            "container": {"padding": "0px", "background-color": "#F5F5F5"},
            "icon": {"color": "#22c55e", "font-size": "18px"},
            "nav-link": {
                "font-family": "Arial, sans-serif",
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "padding": "10px",
                "--hover-color": "#eee",
                "background-color": "white",
                "margin-bottom": "8px",
                "border-radius": "5px",
            },
            "nav-link-selected": {
                "background-color": "#D32F2F",
                "color": "white",
            },
        }
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

# Pages selon la sélection du menu
if selected == "Tableau de bord":
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
                        <div class='metric-title'>Mesure de création moyenne</div>
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
            if 'Taux_Chomage (%)' in filtered_df.columns and 'Year' in filtered_df.columns and 'Sexe' in filtered_df.columns:
             # calcul
                unemployment_trend = (
                    filtered_df
                    .groupby(['Year','Sexe'])['Taux_Chomage (%)']
                    .mean()
                    .reset_index()
                )
                # tracer
                fig = px.line(
                    unemployment_trend,
                    x='Year', y='Taux_Chomage (%)',
                    color='Sexe', markers=True,
                    title="Évolution du taux de chômage par sexe à Perpignan"
                )
                # titres
                fig.update_layout(
                    xaxis_title="Année",
                    yaxis_title="Taux de chômage (%)"
                )
                # n’afficher que des années entières
                fig.update_xaxes(
                    tickmode='linear',
                    dtick=1,
                    tickformat=".0f"
                )
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
        
        # Carte choroplèthe ou carte de densité 
        st.subheader("Distribution géographique des entreprises à Perpignan")
        st.markdown("*Cette carte montre une simulation de la distribution des entreprises dans les différents quartiers de Perpignan.*")
        
        # Exemple de carte simple avec Plotly -
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

elif selected == "Prédiction":
    st.markdown("<h2>Prédiction du taux de chômage</h2>", unsafe_allow_html=True)
    
    if data_loaded:
        st.info("Utilisez ce formulaire pour prédire le taux de chômage en fonction des paramètres socio-économiques.")
        
        # Obtenir l'année minimale pour calculer YearIndex
        min_year = df['Year'].min() if not df.empty else 2017
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Année", min_value=int(min_year), max_value=2030, value=2025)
            # Calculer YearIndex (année - valeur minimum)
            year_index = year - min_year
            employed_total = st.number_input("Nombre total d'employés", min_value=10000, max_value=100000, value=55000)
            population_active = st.number_input("Population active", min_value=10000, max_value=100000, value=60000)
            sexe = st.selectbox("Sexe", ["F", "M"])
            sexe_m = 1 if sexe == "M" else 0
            
        with col2:
            company_count = st.number_input("Nombre d'entreprises", min_value=500, max_value=2000, value=1200)
            avg_cre_measure = st.number_input("Mesure CRE moyenne", min_value=40.0, max_value=60.0, value=50.0)
            age = st.selectbox("Tranche d'âge", ["15-24", "25-39", "40-54", "55-64"])
            # Ajouter un champ supplémentaire pour équilibrer, ou une note explicative
            st.markdown("<div style='margin-top: 16px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'><strong>Note: </strong> Les paramètres ci-dessus influencent directement la prédiction du taux de chômage.</div>", unsafe_allow_html=True)
        
        if st.button("Prédire le taux de chômage", type="primary"):
            try:
                # Préparer les données pour la prédiction avec les transformations nécessaires
                input_data = pd.DataFrame({
                    'YearIndex': [year_index],
                    'Sexe_M': [sexe_m],
                    'age_15-24': [1 if age == "15-24" else 0],
                    'age_25-39': [1 if age == "25-39" else 0],
                    'age_40-54': [1 if age == "40-54" else 0],
                    'age_55-64': [1 if age == "55-64" else 0],
                    'log_Employed_Total': [np.log1p(employed_total)],
                    'log_Population_Active': [np.log1p(population_active)],
                    'log_company_count': [np.log1p(company_count)],
                    'avg_CRE_Measure': [avg_cre_measure]
                })
                
                # S'assurer que toutes les colonnes nécessaires sont présentes dans le bon ordre
                input_data = input_data[feature_info['features']]
                
                # Faire la prédiction
                prediction = model.predict(input_data)[0]

                st.session_state['prediction_result'] = float(prediction)
                st.session_state['prediction_params'] = {
                         "Année": year,
                         "Nombre total d'employés": employed_total,
                          "Population active": population_active,
                          "Sexe": sexe,
                          "Nombre d'entreprises": company_count,
                          "Mesure CRE moyenne": avg_cre_measure,
                          "Tranche d'âge": age
                }

                # Générer le PDF avec les vrais paramètres
                pdf_bytes = export_as_pdf(
                    df=filtered_df,
                    filtered_df=filtered_df,
                    model=model,
                    feature_info=feature_info,
                    foundations_df=foundations_df,
                    state_subv_df=state_subv_df,
                    city_subv_df=city_subv_df,
                    prediction_params=st.session_state['prediction_params'],
                    prediction_result=st.session_state['prediction_result']
                )
                
                # Afficher le bouton de téléchargement
                st.download_button(
                    label="📥 Télécharger le rapport PDF complet",
                    data=pdf_bytes,
                    file_name="rapport_perpignan.pdf",
                    mime="application/pdf"
                )
         
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

elif selected == "Analyse des facteurs":
    st.markdown("<h2>Analyse des facteurs d'influence</h2>", unsafe_allow_html=True)
    
    if data_loaded:
        # Importance des caractéristiques
        st.subheader("Importance des facteurs dans la prédiction du taux de chômage")
        
        # Créer un DataFrame pour l'importance des caractéristiques
        try:
            # Extraire l'importance des features du modèle
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_info['features'],
                'Importance': feature_importances
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
            # Créer des versions non-log pour meilleure lisibilité
            if 'Population_Active' in filtered_df.columns and 'Employed_Total' in filtered_df.columns and 'company_count' in filtered_df.columns:
                correlation_cols = ['Taux_Chomage (%)', 'Employed_Total', 'Population_Active', 'company_count', 'avg_CRE_Measure']
                correlation_cols = [col for col in correlation_cols if col in filtered_df.columns]
                
                if len(correlation_cols) > 1:
                    corr_matrix = filtered_df[correlation_cols].corr()
                    
                    # Afficher la heatmap
                    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                                  title="Matrice de corrélation entre les variables économiques")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Données insuffisantes pour calculer les corrélations")
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
        
        # Ajout d'une nouvelle section pour visualiser l'impact de l'âge
        st.subheader("Impact de la tranche d'âge sur le taux de chômage")
        
        if 'age' in filtered_df.columns and 'Taux_Chomage (%)' in filtered_df.columns:
            age_unemployment = filtered_df.groupby(['age', 'Sexe'])['Taux_Chomage (%)'].mean().reset_index()
            
            fig = px.bar(age_unemployment, x='age', y='Taux_Chomage (%)', 
                       color='Sexe', barmode='group',
                       title="Taux de chômage moyen par tranche d'âge et sexe")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Données insuffisantes pour analyser l'impact de l'âge")
    
    else:
        st.warning("Données non disponibles. Veuillez vérifier le chargement des fichiers.")

#Fondations et subventions page
elif selected == "Fondations et subventions":
    st.markdown("<h2>Fondations et subventions</h2>", unsafe_allow_html=True)
    if data_loaded:

        def render_html_table(df, cols, link_col=None):
            # en-têtes
            html = '<table style="width:100%; border-collapse: collapse; margin-bottom:20px;">'
            html += '<tr style="background-color: #800000;">'
            for c in cols:
                html += (f'<th style="border:1px solid #ddd;'
                         f' padding:8px; text-align:left; color:white;">{c}</th>')
            html += '</tr>'
            # lignes
            for _, row in df.iterrows():
                html += '<tr style="background-color: #f9f9f9;">'
                for c in cols:
                    cell = str(row[c]).strip()
                    if link_col and c == link_col and cell.startswith("http"):
                        html += (f'<td style="border:1px solid #ddd; padding:8px;">'
                                 f'<a href="{cell}" target="_blank">Accéder au site</a></td>')
                    else:
                        html += f'<td style="border:1px solid #ddd; padding:8px;">{cell}</td>'
                html += '</tr>'
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)

        # Fondations européennes
        st.markdown("### Fondations européennes", unsafe_allow_html=True)
        st.markdown("Découvrez ci-dessous la liste des fondations européennes:")
        render_html_table(
            foundations_df,
            cols=["Champs d'intervention", "Fondation ou programme", "Actions"]
        )

        # Subventions étatiques
        st.markdown("### Subventions étatiques", unsafe_allow_html=True)
        st.markdown("Découvrez ci-dessous le tableau des subventions étatiques:")
        render_html_table(
            state_subv_df,
            cols=["Champs d'intervention", "Organisme", "Actions"]
        )

        # Aides et subventions de la Ville de Perpignan
        st.markdown("### Aides et subventions pour la ville de Perpignan", unsafe_allow_html=True)
        st.markdown("Le tableau ci-dessous représente les aides et les subventions de la ville de Perpignan :")
        render_html_table(
            city_subv_df,
            cols=["Projets", "Programmes", "Liens"],
            link_col="Liens"
        )

    else:
        st.warning("Données des fondations non disponibles. Veuillez réessayer plus tard.")
        
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666; margin-top: 30px;">
    © 2025 | Projet de Fin d'Études - Ancrage Territorial de Perpignan | Tous droits réservés
</div>
""", unsafe_allow_html=True)