
# Application Streamlit : Incidence des maladies (Côte d'Ivoire, 2012-2015)

# -------- Importations de packages--------
import streamlit as st                     # Importons Streamlit pour construire l'application web interactive
import pandas as pd                        # Importons pandas pour charger et manipuler les données tabulaires
import numpy as np                         # Importons numpy pour quelques opérations numériques
import plotly.express as px                # Importons Plotly Express pour créer des visualisations interactives
import plotly.graph_objects as go          # Importons Graph Objects pour des graphiques personnalisés
from io import StringIO                    # Importons StringIO pour générer un CSV téléchargeable en mémoire
from sklearn.model_selection import train_test_split, cross_val_score # Importons les outils de découpage/cross-val
from sklearn.compose import ColumnTransformer               # Importons ColumnTransformer pour traiter colonnes hétérogènes
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Importons encodeur catégoriel et standardisation numérique
from sklearn.pipeline import Pipeline                        # Importons Pipeline pour chaîner prétraitements + modèle
from sklearn.linear_model import LinearRegression            # Importons Régression linéaire
from sklearn.preprocessing import PolynomialFeatures         # Importons générateur de caractéristiques polynomiales
from sklearn.ensemble import RandomForestRegressor           # Importons Forêt aléatoire pour la régression
from sklearn.neighbors import KNeighborsRegressor            # Importons KNN régression
from sklearn.neural_network import MLPRegressor             # Importons Perceptron multi-couches (ANN léger, sans TF)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Importons les métriques de régression
import streamlit.components.v1 as components

# -------- Configuration globale de la page --------
st.set_page_config(                             # Configurons la page Streamlit pour un rendu propre
    page_title="Incidence maladies CI (2012-2015)",  # Définissons le titre de l’onglet navigateur
    page_icon="moustique_tigre.jpg",                 # Définissons l’icône de page (image partagée)
    layout="wide"                                    # Passons en mode large pour mieux exploiter l’écran
)

# -------- Styles CSS légers pour homogénéiser l'UI --------
st.markdown("""                                          
<style>
/* Donnons un léger arrondi et des ombres aux blocs */
.block { background: #ffffff; padding: 16px; border-radius: 12px; 
         box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
/* Mettons en valeur les titres secondaires */
h2, h3 { color: #0D1D2C; }
/* Allégeons l’apparence des DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #eee; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --------- Fonctions utilitaires (toutes en français + commentaires ligne par ligne) ---------

@st.cache_data(show_spinner=True)
def charger_donnees_csv(chemin: str) -> pd.DataFrame:
    """Chargeons le fichier CSV pour obtenir le jeu de données source."""
    # Chargeons le fichier CSV pour alimenter toutes les pages de l'application
    df = pd.read_csv(chemin)  # Lecture simple du fichier CSV "Incidence.csv" fourni dans les ressources
    return df                  # Renvoyons le DataFrame chargé

def normaliser_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonisons les libellés de colonnes pour un code robuste, quelles que soient les variantes d'intitulés."""
    # Copions le DataFrame pour éviter les effets de bord
    donnees = df.copy()  # Copions le DataFrame d’entrée pour travailler en sécurité

    # Créons une table de correspondance des libellés hétérogènes vers des noms normalisés (sans espaces/accents)
    # Nous couvrons les deux variantes vues dans tes fichiers : avec slash/espaces et avec underscore.
    mapping = {
        "ANNEE": "annee",
        "REGIONS / DISTRICTS": "regions_districts",
        "REGIONS/DISTRICTS": "regions_districts",
        "VILLES / COMMUNES": "villes_communes",
        "VILLES/COMMUNES": "villes_communes",
        "MALADIE": "maladie",
        "INCIDENCE SUR LA POPULATION GENERALE (%)": "incidence_population_pct",
        "INCIDENCE_SUR_LA_POPULATION_GENERALE_(%)": "incidence_population_pct",
    }

    # Renommons les colonnes existantes si elles figurent dans le mapping
    colonnes_renommees = {c: mapping[c] for c in donnees.columns if c in mapping}  # Préparons le dict des colonnes présentes
    donnees = donnees.rename(columns=colonnes_renommees)  # Appliquons le renommage

    # Pour plus de robustesse, normalisons tout le reste (minuscules + remplaçons espaces par underscore)
    donnees.columns = [c.strip().lower().replace(" ", "_") for c in donnees.columns]  # Nettoyons tous les noms de colonnes
    return donnees  # Renvoyons le DataFrame aux noms standardisés

def typer_colonnes(donnees: pd.DataFrame) -> pd.DataFrame:
    """Typage : convertissons année en entier, incidence en float, autres en str pour des traitements cohérents."""
    df = donnees.copy()                                   # Copions le DataFrame
    if "annee" in df.columns:                             # Vérifions la présence de la colonne annee
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")  # Convertissons en entier tolérant NA
    if "incidence_population_pct" in df.columns:          # Vérifions la présence de la colonne d’incidence
        df["incidence_population_pct"] = pd.to_numeric(df["incidence_population_pct"], errors="coerce")  # Convertissons en float
    # Forçons les colonnes catégorielles en string pour éviter les surprises plus tard
    for col in ["regions_districts", "villes_communes", "maladie"]:
        if col in df.columns:
            df[col] = df[col].astype("string")            # Convertissons en chaînes (type pandas string)
    return df                                             # Renvoyons le DataFrame typé

def statistiques_rapides(df: pd.DataFrame) -> pd.DataFrame:
    """Produisons un tableau synthétique des statistiques descriptives numériques pour survol rapide."""
    # Utilisons describe() pour obtenir n, moyenne, std, min, max, quartiles sur les colonnes numériques
    stats = df.select_dtypes(include=[np.number]).describe().T  # Calculons les statistiques et transposons pour lisibilité
    return stats                                           # Renvoyons le tableau de stats

def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    """Comptons les valeurs manquantes par colonne pour cibler le nettoyage."""
    # Sommons les NA par colonne et trions par décroissant
    na = df.isna().sum().sort_values(ascending=False)     # Comptons les NA par colonne
    return pd.DataFrame({"colonne": na.index, "manquants": na.values})  # Renvoyons un DataFrame propre

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyons : supprimons les doublons, imputons les NA (mode sur catégoriel, médiane sur numérique)."""
    donnees = df.copy()                                    # Copions pour préserver l’original
    donnees = donnees.drop_duplicates()                    # Supprimons les éventuelles lignes dupliquées

    # Séparons colonnes numériques et catégorielles
    cols_num = donnees.select_dtypes(include=[np.number]).columns.tolist()  # Repérons les colonnes numériques
    cols_cat = [c for c in donnees.columns if c not in cols_num]            # Les autres seront traitées comme catégorielles

    # Imputons les valeurs manquantes numériques par la médiane (robuste aux outliers)
    for c in cols_num:
        if donnees[c].isna().any():                        # Si la colonne contient des NA
            donnees[c] = donnees[c].fillna(donnees[c].median())  # Remplaçons par la médiane

    # Imputons les valeurs manquantes catégorielles par le mode (valeur la plus fréquente)
    for c in cols_cat:
        if donnees[c].isna().any():                        # Si la colonne contient des NA
            mode = donnees[c].mode(dropna=True)            # Calculons la modalité dominante
            if len(mode) > 0:                              # Vérifions que le mode existe
                donnees[c] = donnees[c].fillna(mode.iloc[0])  # Remplaçons par cette modalité

    return donnees                                         # Renvoyons le DataFrame nettoyé

def detecter_valeurs_aberrantes(df: pd.DataFrame, z=3.0) -> pd.DataFrame:
    """Détectons les valeurs aberrantes (z-score > seuil) pour les colonnes numériques."""
    # Sélectionnons uniquement les colonnes numériques
    dnum = df.select_dtypes(include=[np.number])           # Extrayons les colonnes numériques
    # Calculons le z-score absolu et marquons les lignes contenant au moins un dépassement
    zscores = (dnum - dnum.mean()) / dnum.std(ddof=0)      # Calculons les z-scores (écart-type population)
    masque_out = (zscores.abs() > z).any(axis=1)           # Identifions les lignes avec au moins un z-score > seuil
    return df.loc[masque_out]                              # Renvoyons uniquement les lignes suspectes

def telecharger_csv(df: pd.DataFrame) -> bytes:
    """Préparons un CSV téléchargeable (en mémoire) pour récupérer les données nettoyées."""
    buffer = StringIO()                                    # Ouvrons un tampon mémoire texte
    df.to_csv(buffer, index=False)                         # Écrivons le CSV dans le tampon
    return buffer.getvalue().encode("utf-8")               # Renvoyons les octets encodés en UTF-8

# --------- Chargement unique & préparation initiale ---------
# Chargeons le fichier 'Incidence.csv' pour alimenter l'intégralité de l'app, puis normalisons/typons/Nettoyons
donnees_brutes = charger_donnees_csv("Incidence.csv")          # Chargeons le fichier CSV pour obtenir les données d'origine
donnees = normaliser_noms_colonnes(donnees_brutes)             # Normalisons les libellés pour unifier le code
donnees = typer_colonnes(donnees)                              # Typage cohérent (entiers, float, string)
donnees_nettoyees = nettoyer_donnees(donnees)                  # Appliquons un nettoyage simple et robuste

# Stockons dans la session pour réutiliser partout sans rechargement
if "donnees_nettoyees" not in st.session_state:                # Vérifions si la session contient déjà les données
    st.session_state["donnees_nettoyees"] = donnees_nettoyees  # Déposons les données nettoyées dans la session

# --------- Barre de navigation horizontale (onglets) ---------
onglets = st.tabs([                                           # Créons des onglets pour une navigation horizontale claire
    "🏠 Accueil", "📒 Informations", "🛠 Exploration", "🧹 Préparation",
    "🔍 Visualisations", "👀 Explorateur", "〽️ Modélisation", "◻ Prédiction", "🛖 Source"
])

# =========================
# 🏠 ACCUEIL
# =========================
with onglets[0]:
    # Affichons un en-tête agréable
    st.title("Incidence des maladies en Côte d’Ivoire (2012–2015)")  # Titre principal de l'application
    col1, col2 = st.columns([1, 2], gap="large")                     # Découpons en deux colonnes pour la mise en page

    with col1:
        st.image("moustique_tigre.jpg", use_column_width=True, caption="Aedes albopictus (moustique tigre)")  # Affichons l'image fournie

    with col2:
        st.markdown("""<div class="block" style="text-align:justify">
        <h3>Objectif de l’application</h3>
        Cette application interactive permet d’explorer, de visualiser et de modéliser 
        l’incidence de plusieurs maladies en Côte d’Ivoire entre 2012 et 2015.
        Elle propose des graphiques interactifs, un explorateur visuel libre (Pygwalker),
        ainsi que plusieurs modèles prédictifs (Régression, Random Forest, KNN, ANN).
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="block" style="text-align:justify">
        <h3>Problème adressé</h3>
        Comment transformer des données brutes de santé publique en <b>indicateurs actionnables</b> 
        et en <b>prédictions</b> fiables pour aider à la décision (priorisation des zones et des pathologies) ?
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="block" style="text-align:justify">
        <h3>Résultats attendus</h3>
        Un <b>outil unique</b> permettant : 
        (1) l’analyse rapide des tendances,
        (2) l’identification des disparités régionales,
        (3) la prédiction de l’incidence selon année/région/ville/maladie.
        </div>""", unsafe_allow_html=True)

# =========================
# 📒 INFORMATIONS
# =========================
with onglets[1]:
    st.header("Informations sur les données")                      # Plaçons un en-tête clair
    st.write("**Aperçu des premières lignes (jeu de données d’origine)**")  # Introduisons l’aperçu
    st.dataframe(donnees_brutes.head(), use_container_width=True)  # Affichons les 5 premières lignes d'origine

    st.write("**Libellés de colonnes normalisés (utilisés en interne)**")   # Expliquons les noms normalisés
    st.json({
        "annee": "Année d’observation (2012–2015)",
        "regions_districts": "Région ou district sanitaire",
        "villes_communes": "Ville ou commune",
        "maladie": "Type de pathologie",
        "incidence_population_pct": "Incidence sur la population générale (en %)"
    })

# =========================
# 🛠 EXPLORATION
# =========================
with onglets[2]:
    st.header("Exploration des données")                               # Titre de section
    colA, colB = st.columns(2)                                         # Créons 2 colonnes

    with colA:
        st.subheader("Dimensions & dtypes")                             # Sous-titre
        st.write(f"**Nombre de lignes** : {donnees_nettoyees.shape[0]}")# Affichons nb lignes
        st.write(f"**Nombre de colonnes** : {donnees_nettoyees.shape[1]}") # Affichons nb colonnes
        st.write("**Types de données**")                                # Présentons les types
        st.dataframe(donnees_nettoyees.dtypes.to_frame("dtype"), use_container_width=True)  # Montrons les dtypes

    with colB:
        st.subheader("Valeurs manquantes")                              # Sous-titre
        st.dataframe(valeurs_manquantes(donnees_nettoyees), use_container_width=True)  # Affichons NA par colonne

    st.subheader("Statistiques descriptives (variables numériques)")     # Sous-titre pour stats
    st.dataframe(statistiques_rapides(donnees_nettoyees), use_container_width=True)     # Affichons describe()

    st.subheader("Valeurs aberrantes potentielles (z-score > 3)")       # Sous-titre pour outliers
    outliers = detecter_valeurs_aberrantes(donnees_nettoyees)           # Détectons les valeurs anormales
    if outliers.empty:
        st.info("Aucune ligne ne dépasse le seuil de z-score sélectionné (3.0).")       # Message si rien
    else:
        st.dataframe(outliers, use_container_width=True)                # Affichons les lignes suspectes

# =========================
# 🧹 PRÉPARATION (manipulation)
# =========================
with onglets[3]:
    st.header("Préparation et export des données")                      # Titre de section

    st.write("**Aperçu après nettoyage (doublons supprimés, NA imputés)**")  # Introduisons l’aperçu post-nettoyage
    st.dataframe(donnees_nettoyees.head(20), use_container_width=True)       # Affichons 20 lignes pour contrôle

    # Bouton de téléchargement du CSV nettoyé
    st.download_button(                                                  # Créons un bouton pour récupérer le CSV propre
        label="📥 Télécharger les données nettoyées (CSV)",              # Étiquette du bouton
        data=telecharger_csv(donnees_nettoyees),                         # Contenu du fichier en mémoire
        file_name="incidence_nettoyee.csv",                              # Nom du fichier proposé
        mime="text/csv"                                                  # Type MIME
    )

# =========================
# 🔍 VISUALISATIONS
# =========================
with onglets[4]:
    st.header("Visualisations interactives")                             # Titre de section
    dfv = donnees_nettoyees                                              # Alias local pour alléger l'écriture

    # ----- Contrôles d’entrée pour filtrer les graphiques -----
    colf1, colf2, colf3 = st.columns(3)                                  # Trois filtres côte à côte

    with colf1:
        maladies_dispo = sorted(dfv["maladie"].dropna().unique().tolist())   # Récupérons la liste des maladies
        choix_maladie = st.selectbox("Choisir une maladie", maladies_dispo, index=0)  # Sélecteur maladie

    with colf2:
        annees_dispo = sorted(dfv["annee"].dropna().unique().astype(int).tolist())    # Récupérons la liste des années
        choix_annees = st.multiselect("Filtrer par année", annees_dispo, default=annees_dispo) # Sélecteur multi années

    with colf3:
        regions_dispo = ["(Toutes)"] + sorted(dfv["regions_districts"].dropna().unique().tolist()) # Liste des régions
        choix_region = st.selectbox("Filtrer par région/district", regions_dispo, index=0)         # Sélecteur région

    # ----- Appliquons les filtres -----
    dff = dfv[dfv["maladie"] == choix_maladie]                                 # Filtrons sur la maladie sélectionnée
    dff = dff[dff["annee"].isin(choix_annees)]                                 # Filtrons sur les années sélectionnées
    if choix_region != "(Toutes)":                                             # Si une région précise est choisie
        dff = dff[dff["regions_districts"] == choix_region]                    # Filtrons sur la région choisie

    # ----- Graphique 1 : évolution temporelle -----
    st.subheader("Évolution de l’incidence (%) dans le temps")                 # Sous-titre
    if len(dff) == 0:                                                          # Si le filtre est trop restrictif
        st.warning("Aucune donnée pour ce filtre.")                            # Avertissons l'utilisateur
    else:
        # Calculons la moyenne par année pour lisser les valeurs
        evol = dff.groupby("annee", dropna=True)["incidence_population_pct"].mean().reset_index()  # Moyenne par année
        fig1 = px.line(evol, x="annee", y="incidence_population_pct",
                       markers=True, labels={"annee": "Année", "incidence_population_pct": "Incidence (%)"},
                       title=f"Évolution — {choix_maladie}")                   # Construisons une courbe avec marqueurs
        st.plotly_chart(fig1, use_container_width=True)                        # Affichons le graphique

    # ----- Graphique 2 : comparaison par région (moyenne) -----
    st.subheader("Comparaison des régions/districts (moyenne)")               # Sous-titre
    comp = dfv[dfv["maladie"] == choix_maladie]                                # Filtrons sur la maladie
    comp = comp[comp["annee"].isin(choix_annees)]                              # Filtrons sur les années
    comp = comp.groupby("regions_districts", dropna=True)["incidence_population_pct"].mean().reset_index()  # Moyennes
    comp = comp.sort_values("incidence_population_pct", ascending=False).head(20)  # Gardons les 20 plus élevées pour lisibilité
    fig2 = px.bar(comp, x="regions_districts", y="incidence_population_pct",
                  labels={"regions_districts": "Région/District", "incidence_population_pct": "Incidence (%)"},
                  title=f"Top régions — {choix_maladie} (moyenne {min(choix_annees)}–{max(choix_annees)})") # Barres triées
    fig2.update_layout(xaxis_tickangle=-45)                                    # Penchons les étiquettes pour lisibilité
    st.plotly_chart(fig2, use_container_width=True)                            # Affichons

    # ----- Graphique 3 : répartition par année (camembert) -----
    st.subheader("Répartition par année (moyenne)")                            # Sous-titre
    rep = dff.groupby("annee", dropna=True)["incidence_population_pct"].mean().reset_index()      # Moyenne par année
    fig3 = px.pie(rep, names="annee", values="incidence_population_pct",
                  title="Part relative moyenne par année", hole=0.35)          # Camembert en donut
    st.plotly_chart(fig3, use_container_width=True)                            # Affichons

# =========================
# 👀 EXPLORATEUR (Pygwalker)
# =========================
with onglets[5]:
    st.header("Explorateur visuel libre (Pygwalker)")
    st.info("Astuce : glissez-déposez les champs à gauche pour créer vos vues interactives.")

    try:
        # Chargeons Pygwalker pour générer un studio d’exploration visuelle embarqué
        import pygwalker as pyg

        # Convertissons le DataFrame en interface HTML interactive de Pygwalker
        html = pyg.to_html(donnees_nettoyees)

        # ✅ Appel correct de l’API Streamlit Components
        components.html(html, height=900, scrolling=True)

    except ModuleNotFoundError:
        st.error("Pygwalker n’est pas installé. Vérifiez qu’il est bien dans requirements.txt (pygwalker==0.4.8.9).")
    except Exception as e:
        st.error("Pygwalker n’a pas pu être chargé. Vérifiez l’environnement de déploiement.")
        st.exception(e)

# =========================
# 〽️ MODÉLISATION
# =========================
with onglets[6]:
    st.header("Modélisation supervisée (régression)")                          # Titre de section

    # --------- Paramètres utilisateur ----------
    colp1, colp2, colp3 = st.columns(3)                                        # Trois colonnes de paramètres modèle
    with colp1:
        type_modele = st.selectbox("Choisir un modèle",
                                   ["Régression linéaire", "Régression polynomiale", "Forêt aléatoire",
                                    "KNN régression", "Réseau de neurones (ANN)"])  # Choix du modèle
    with colp2:
        test_size = st.slider("Taille test (%)", 10, 40, 20, step=5)           # Pourcentage du jeu de test
    with colp3:
        random_state = st.number_input("Graine aléatoire", value=42, step=1)   # Graine pour reproductibilité

    # --------- Préparation X / y ----------
    # Chargeons les caractéristiques pour prédire l'incidence (%)
    X = donnees_nettoyees[["annee", "regions_districts", "villes_communes", "maladie"]]  # Sélectionnons les features
    y = donnees_nettoyees["incidence_population_pct"]                                     # Définissons la cible

    # Définissons colonnes numériques et catégorielles
    colonnes_numeriques = ["annee"]                                                       # Une seule numérique
    colonnes_categorielles = ["regions_districts", "villes_communes", "maladie"]          # Trois catégorielles

    # Construisons le préprocesseur (standardisation + one-hot)
    preprocesseur = ColumnTransformer(                                                    # Créons un transformateur par type
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),                               # Standardisons les numériques
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles)       # Encodons en one-hot les catégorielles
        ]
    )

    # Décomposons les données en apprentissage/test
    X_train, X_test, y_train, y_test = train_test_split(                                  # Effectuons le split en un appel
        X, y, test_size=test_size/100, random_state=int(random_state)
    )

    # --------- Sélection/Construction du pipeline modèle ----------
    if type_modele == "Régression linéaire":                                              # Cas 1 : régression linéaire
        modele = LinearRegression()                                                       # Instancions le modèle simple
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])                   # Chaignons prétraitement + modèle

    elif type_modele == "Régression polynomiale":                                         # Cas 2 : régression polynomiale
        # Ajoutons des termes polynomiaux (sur les variables numériques déjà standardisées)
        # NB : On applique PolynomialFeatures après l'encodage via une sous-pipeline pour n’agir que sur les num.
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),                   # Créons des interactions quadratiques
            ("mod", LinearRegression())                                                   # Finissons par une régression linéaire
        ])

    elif type_modele == "Forêt aléatoire":                                                # Cas 3 : RandomForest Regressor
        modele = RandomForestRegressor(n_estimators=300, random_state=int(random_state))  # Initialisons la forêt
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])                   # Chaignons avec le prétraitement

    elif type_modele == "KNN régression":                                                 # Cas 4 : KNN
        modele = KNeighborsRegressor(n_neighbors=7)                                       # Instancions un KNN (k=7 par défaut)
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])                   # Pipeline complet

    else:                                                                                 # Cas 5 : ANN via MLPRegressor
        modele = MLPRegressor(hidden_layer_sizes=(128, 64),                               # Réseau 2 couches (128, 64)
                              activation="relu", solver="adam",
                              max_iter=1000, random_state=int(random_state))
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])                   # Pipeline complet

    # --------- Entraînement + Évaluation ----------
    # Entraînons le pipeline sur X_train, y_train
    pipeline.fit(X_train, y_train)                                                        # Apprenons le modèle sur le jeu train

    # Prédictions sur X_test
    y_pred = pipeline.predict(X_test)                                                     # Produisons les prédictions de test

    # Calculons les métriques
    r2 = r2_score(y_test, y_pred)                                                         # Calculons le R²
    mae = mean_absolute_error(y_test, y_pred)                                             # Erreur absolue moyenne
    rmse = mean_squared_error(y_test, y_pred, squared=False)                              # Racine de l’erreur quadratique

    colm1, colm2, colm3 = st.columns(3)                                                   # Trois cartes de métriques
    colm1.metric("R²", f"{r2:0.3f}")                                                      # Affichons le R²
    colm2.metric("MAE", f"{mae:0.3f}")                                                    # Affichons le MAE
    colm3.metric("RMSE", f"{rmse:0.3f}")                                                  # Affichons le RMSE

    # Cross-validation 5-fold pour robustesse (sur l’ensemble complet)
    with st.expander("Voir la validation croisée (5 plis)"):
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")                      # Calculons les R² en CV
        st.write("Scores R² par pli :", [f"{s:0.3f}" for s in scores])                    # Affichons les scores par pli
        st.write("R² moyen :", f"{np.mean(scores):0.3f} ± {np.std(scores):0.3f}")        # Moyenne et écart-type

    # Mettons le pipeline dans la session pour réutilisation dans l’onglet Prédiction
    st.session_state["pipeline_modele"] = pipeline                                        # Stockons le pipeline entraîné
    st.success("Modèle entraîné et stocké pour l’onglet ◻ Prédiction.")                   # Indiquons la disponibilité

# =========================
# ◻ PRÉDICTION
# =========================
with onglets[7]:
    st.header("Prédire l’incidence (%) selon vos sélections")                             # Titre de section

    if "pipeline_modele" not in st.session_state:                                         # Vérifions qu’un modèle existe
        st.warning("Veuillez d’abord entraîner un modèle dans l’onglet 〽️ Modélisation.")
    else:
        pipe = st.session_state["pipeline_modele"]                                        # Récupérons le pipeline

        colpr1, colpr2, colpr3, colpr4 = st.columns(4)                                    # Quatre critères d’entrée
        with colpr1:
            annee_sel = st.selectbox("Année", sorted(donnees_nettoyees["annee"].dropna().astype(int).unique()))
        with colpr2:
            region_sel = st.selectbox("Région/District", sorted(donnees_nettoyees["regions_districts"].dropna().unique()))
        with colpr3:
            ville_sel = st.selectbox("Ville/Commune", sorted(donnees_nettoyees["villes_communes"].dropna().unique()))
        with colpr4:
            maladie_sel = st.selectbox("Maladie", sorted(donnees_nettoyees["maladie"].dropna().unique()))

        # Construisons un DataFrame d’une ligne avec ces choix
        saisie = pd.DataFrame({
            "annee": [int(annee_sel)],
            "regions_districts": [region_sel],
            "villes_communes": [ville_sel],
            "maladie": [maladie_sel]
        })

        if st.button("🔮 Lancer la prédiction"):
            # Produisons la prédiction via le pipeline entraîné
            y_hat = pipe.predict(saisie)[0]                                              # Récupérons la prédiction scalaire
            st.success(f"Incidence attendue : **{y_hat:.2f} %**")                        # Affichons un résumé clair

            # Optionnel : rapprochons de la moyenne historique locale comme repère
            cond = (
                (donnees_nettoyees["annee"] == int(annee_sel)) &
                (donnees_nettoyees["regions_districts"] == region_sel) &
                (donnees_nettoyees["villes_communes"] == ville_sel) &
                (donnees_nettoyees["maladie"] == maladie_sel)
            )
            ref_local = donnees_nettoyees.loc[cond, "incidence_population_pct"].mean()   # Calculons une moyenne de référence
            if not np.isnan(ref_local):
                st.info(f"Moyenne observée (mêmes filtres, historique) : **{ref_local:.2f} %**")

# =========================
#  SOURCE
# =========================
with onglets[8]:
    st.header("Origine des données")                                                      # Titre de section
    st.markdown("""
    - **Source ouverte** : *Incidences de maladies sur la population de 2012 à 2015*.  
    - **Lien** : https://data.gouv.ci/datasets/incidence-de-maladies-sur-la-population-de-2012-a-2015
    """)
    st.write("Utilisez les autres onglets pour naviguer dans l’analyse, la modélisation et la prédiction.")
