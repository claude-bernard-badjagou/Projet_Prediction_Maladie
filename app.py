
# Application Streamlit : Incidence des maladies (Côte d'Ivoire, 2012-2015)


# -------- Imports (tous commentés) --------
import streamlit as st                                 # Application web interactive
import streamlit.components.v1 as components           # Composants HTML (intégration Pygwalker)
import pandas as pd                                    # Manipulation de données tabulaires
import numpy as np                                     # Calcul numérique
import plotly.express as px                            # Visualisations interactives
import plotly.graph_objects as go                      # Graphiques personnalisés
from io import StringIO                                # Tampon texte pour export CSV

# Outils ML
from sklearn.model_selection import train_test_split, cross_val_score  # Split + CV
from sklearn.compose import ColumnTransformer                           # Prétraitement hétérogène
from sklearn.preprocessing import OneHotEncoder, StandardScaler         # Encodage + standardisation
from sklearn.pipeline import Pipeline                                   # Pipeline prétraitement+modèle
from sklearn.linear_model import LinearRegression                       # Régression linéaire
from sklearn.preprocessing import PolynomialFeatures                    # Caractéristiques polynomiales
from sklearn.ensemble import RandomForestRegressor                      # Forêt aléatoire
from sklearn.neighbors import KNeighborsRegressor                       # KNN
from sklearn.neural_network import MLPRegressor                         # Réseau de neurones (MLP)
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error  # Métriques

# -------- Options pandas (prévenir les downcasting silencieux) --------
pd.set_option('future.no_silent_downcasting', True)  # Évite les changements silencieux de dtype futurs

# -------- Configuration globale de la page --------
st.set_page_config(
    page_title="Incidence maladies CI (2012-2015)",   # Titre de l’onglet navigateur
    page_icon="moustique_tigre.jpg",                  # Icône de page
    layout="wide"                                     # Mise en page large
)

# -------- Styles CSS légers pour homogénéiser l'UI --------
st.markdown("""
<style>
.block { background:#fff; padding:16px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
h2, h3 { color:#0D1D2C; }
[data-testid="stDataFrame"] { border:1px solid #eee; border-radius:8px; }
.section { margin-top:16px; margin-bottom:24px; }
</style>
""", unsafe_allow_html=True)

# --------- Fonctions utilitaires (toutes en français + commentaires) ---------

@st.cache_data(show_spinner=True)
def charger_donnees_csv(chemin: str) -> pd.DataFrame:
    """Chargeons le fichier CSV pour obtenir le jeu de données source (unique point d'entrée)."""
    df = pd.read_csv(chemin)                      # Lecture du fichier CSV "Incidence.csv"
    return df                                     # Renvoyons les données brutes

def normaliser_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonisons les libellés de colonnes pour un code robuste, malgré les variantes d'intitulés."""
    donnees = df.copy()                           # Copions le DataFrame pour travailler proprement
    mapping = {                                   # Table de correspondance des noms hétérogènes
        "ANNEE": "annee",
        "REGIONS / DISTRICTS": "regions_districts",
        "REGIONS/DISTRICTS": "regions_districts",
        "VILLES / COMMUNES": "villes_communes",
        "VILLES/COMMUNES": "villes_communes",
        "MALADIE": "maladie",
        "INCIDENCE SUR LA POPULATION GENERALE (%)": "incidence_population_pct",
        "INCIDENCE_SUR_LA_POPULATION_GENERALE_(%)": "incidence_population_pct",
    }
    donnees = donnees.rename(columns={c: mapping[c] for c in donnees.columns if c in mapping})  # Renommage ciblé
    donnees.columns = [c.strip().lower().replace(" ", "_") for c in donnees.columns]            # Normalisation globale
    return donnees

def typer_colonnes(donnees: pd.DataFrame) -> pd.DataFrame:
    """Typage : année en float64 (compatible Arrow), incidence en float64, autres en string."""
    df = donnees.copy()
    if "annee" in df.columns:
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce")                   # Année -> float64 (gère NaN)
    if "incidence_population_pct" in df.columns:
        df["incidence_population_pct"] = pd.to_numeric(df["incidence_population_pct"], errors="coerce")  # Float64
    for col in ["regions_districts", "villes_communes", "maladie"]:
        if col in df.columns:
            df[col] = df[col].astype("string")                                      # Catégories -> string pandas
    return df

def statistiques_rapides(df: pd.DataFrame) -> pd.DataFrame:
    """Produisons un tableau synthétique des statistiques descriptives numériques."""
    return df.select_dtypes(include=[np.number]).describe().T

def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    """Comptons les valeurs manquantes par colonne pour cibler le nettoyage."""
    na = df.isna().sum().sort_values(ascending=False)
    return pd.DataFrame({"colonne": na.index, "manquants": na.values})

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyons : supprimons les doublons, imputons NA (médiane sur numérique, mode sur catégoriel)."""
    donnees = df.copy().drop_duplicates()                                             # Supprimons les doublons
    cols_num = donnees.select_dtypes(include=[np.number]).columns.tolist()            # Numériques
    cols_cat = [c for c in donnees.columns if c not in cols_num]                      # Catégorielles
    for c in cols_num:                                                                
        if donnees[c].isna().any():                                                   # Imputation numérique
            donnees[c] = donnees[c].fillna(donnees[c].median())
    for c in cols_cat:
        if donnees[c].isna().any():                                                   # Imputation catégorielle
            mode = donnees[c].mode(dropna=True)
            if len(mode) > 0:
                donnees[c] = donnees[c].fillna(mode.iloc[0])
    return donnees

def detecter_valeurs_aberrantes(df: pd.DataFrame, z=3.0) -> pd.DataFrame:
    """Détectons les valeurs aberrantes (z-score > seuil) pour les colonnes numériques."""
    dnum = df.select_dtypes(include=[np.number])
    if dnum.empty:
        return df.iloc[0:0]
    zscores = (dnum - dnum.mean()) / dnum.std(ddof=0)                                 # Z-scores (écart-type population)
    masque_out = (zscores.abs() > z).any(axis=1)                                      # Lignes avec au moins un dépassement
    return df.loc[masque_out]

def telecharger_csv(df: pd.DataFrame) -> bytes:
    """Préparons un CSV téléchargeable (en mémoire) pour récupérer les données nettoyées."""
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

def rendre_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertissons les types pandas potentiellement problématiques (nullable, objets mixtes)
    vers des types Arrow-compatibles avant affichage Streamlit, sans downcasting silencieux.
    """
    dfa = df.copy()
    dfa = dfa.replace({pd.NA: np.nan})                     # Remplaçons explicitement pd.NA par np.nan
    dfa.infer_objects(copy=False)                          # Alignons les dtypes object -> types concrets (sans downcasting silencieux)
    # Colonnes 'object' hétérogènes -> string si mélange de types
    for col in dfa.columns:
        if dfa[col].dtype == "object":
            types_uniques = set(type(x) for x in dfa[col].dropna().head(1000))
            if len(types_uniques) > 1:
                dfa[col] = dfa[col].astype("string")
    return dfa

# --------- Chargement unique & préparation initiale ---------
# Chargeons le fichier 'Incidence.csv' pour alimenter l'app, puis normalisons/typons/nettoyons
donnees_brutes = charger_donnees_csv("Incidence.csv")          # Chargeons le fichier source
donnees = normaliser_noms_colonnes(donnees_brutes)             # Uniformisons les noms de colonnes
donnees = typer_colonnes(donnees)                              # Typage Arrow-friendly
donnees_nettoyees = nettoyer_donnees(donnees)                  # Nettoyage simple & robuste

# Stockons dans la session pour réutiliser partout
if "donnees_nettoyees" not in st.session_state:
    st.session_state["donnees_nettoyees"] = donnees_nettoyees

# --------- (Option) Auto-entraînement au démarrage (robuste) ---------
AUTO_TRAIN_ON_START = True  # Mets False si tu préfères entraîner manuellement dans l’onglet Modélisation

def construire_pipeline_defaut():
    """Pipeline par défaut (prétraitements + RandomForest)."""
    colonnes_numeriques = ["annee"]
    colonnes_categorielles = ["regions_districts", "villes_communes", "maladie"]
    preprocesseur = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
        ]
    )
    modele = RandomForestRegressor(n_estimators=300, random_state=42)
    return Pipeline([("prep", preprocesseur), ("mod", modele)])

def entrainer_au_demarrage_si_absent(df: pd.DataFrame):
    """
    Entraîne un modèle de base si absent.
    - Supprime les lignes incomplètes (dropna) pour éviter un crash.
    - Capture les exceptions pour ne pas bloquer le rendu des onglets.
    """
    try:
        data = df[["annee", "regions_districts", "villes_communes", "maladie", "incidence_population_pct"]].dropna()
        if data.empty:
            st.warning("Auto-entraînement ignoré : données insuffisantes (trop de valeurs manquantes).")
            return
        X = data[["annee", "regions_districts", "villes_communes", "maladie"]]
        y = data["incidence_population_pct"]
        pipe = construire_pipeline_defaut()
        pipe.fit(X, y)
        st.session_state["pipeline_modele"] = pipe
        st.session_state["modele_info"] = "Modèle par défaut (RandomForest) entraîné au démarrage."
    except Exception as e:
        st.warning("Auto-entraînement au démarrage non réalisé (erreur capturée). Consulte l’onglet 〽️ Modélisation.")
        st.caption(f"Détail : {type(e).__name__}: {e}")

if AUTO_TRAIN_ON_START and "pipeline_modele" not in st.session_state:
    entrainer_au_demarrage_si_absent(st.session_state["donnees_nettoyees"])

# --------- Barre de navigation horizontale (onglets) ---------
onglets = st.tabs([
    "🏠 Accueil", "📒 Informations", "🛠 Exploration", "🧹 Préparation",
    "🔍 Visualisations", "👀 Explorateur", "〽️ Modélisation", "◻ Prédiction", "🛖 Source"
])

# =========================
# 🏠 ACCUEIL
# =========================
with onglets[0]:
    st.title("Incidence des maladies en Côte d’Ivoire (2012–2015)")
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.image("moustique_tigre.jpg", use_column_width=True, caption="Aedes albopictus (moustique tigre)")
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
    st.header("Informations sur les données")
    st.write("**Aperçu des premières lignes (jeu de données d’origine)**")
    st.dataframe(rendre_arrow_compatible(donnees_brutes.head()), use_container_width=True)

    st.write("**Libellés de colonnes normalisés (utilisés en interne)**")
    st.json({
        "annee": "Année d’observation (2012–2015)",
        "regions_districts": "Région ou district sanitaire",
        "villes_communes": "Ville ou commune",
        "maladie": "Type de pathologie",
        "incidence_population_pct": "Incidence sur la population générale (en %)"
    })

# =========================
# 🛠 EXPLORATION  (tables empilées verticalement)
# =========================
with onglets[2]:
    st.header("Exploration des données")

    st.subheader("Dimensions")
    st.write(f"**Nombre de lignes** : {donnees_nettoyees.shape[0]}")
    st.write(f"**Nombre de colonnes** : {donnees_nettoyees.shape[1]}")

    st.subheader("Types de données (dtypes)")
    st.dataframe(rendre_arrow_compatible(donnees_nettoyees.dtypes.to_frame("dtype")), use_container_width=True)

    st.subheader("Valeurs manquantes (par colonne)")
    st.dataframe(rendre_arrow_compatible(valeurs_manquantes(donnees_nettoyees)), use_container_width=True)

    st.subheader("Statistiques descriptives (variables numériques)")
    st.dataframe(rendre_arrow_compatible(statistiques_rapides(donnees_nettoyees)), use_container_width=True)

    st.subheader("Valeurs aberrantes potentielles (z-score > 3)")
    outliers = detecter_valeurs_aberrantes(donnees_nettoyees)
    if outliers.empty:
        st.info("Aucune ligne ne dépasse le seuil de z-score sélectionné (3.0).")
    else:
        st.dataframe(rendre_arrow_compatible(outliers), use_container_width=True)

# =========================
# 🧹 PRÉPARATION (manipulation)
# =========================
with onglets[3]:
    st.header("Préparation et export des données")
    st.write("**Aperçu après nettoyage (doublons supprimés, NA imputés)**")
    st.dataframe(rendre_arrow_compatible(donnees_nettoyees.head(20)), use_container_width=True)
    st.download_button(
        label="📥 Télécharger les données nettoyées (CSV)",
        data=telecharger_csv(donnees_nettoyees),
        file_name="incidence_nettoyee.csv",
        mime="text/csv"
    )

# =========================
# 🔍 VISUALISATIONS
# =========================
with onglets[4]:
    st.header("Visualisations interactives")
    dfv = donnees_nettoyees

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        maladies_dispo = sorted(dfv["maladie"].dropna().unique().tolist())
        choix_maladie = st.selectbox("Choisir une maladie", maladies_dispo, index=0)
    with colf2:
        annees_dispo = sorted(dfv["annee"].dropna().unique().astype(int).tolist())
        choix_annees = st.multiselect("Filtrer par année", annees_dispo, default=annees_dispo)
    with colf3:
        regions_dispo = ["(Toutes)"] + sorted(dfv["regions_districts"].dropna().unique().tolist())
        choix_region = st.selectbox("Filtrer par région/district", regions_dispo, index=0)

    # Filtrage
    dff = dfv[dfv["maladie"] == choix_maladie]
    dff = dff[dff["annee"].isin(choix_annees)]
    if choix_region != "(Toutes)":
        dff = dff[dff["regions_districts"] == choix_region]

    st.subheader("Évolution de l’incidence (%) dans le temps")
    if len(dff) == 0:
        st.warning("Aucune donnée pour ce filtre.")
    else:
        evol = dff.groupby("annee", dropna=True)["incidence_population_pct"].mean().reset_index()
        fig1 = px.line(
            evol, x="annee", y="incidence_population_pct", markers=True,
            labels={"annee": "Année", "incidence_population_pct": "Incidence (%)"},
            title=f"Évolution — {choix_maladie}"
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Comparaison des régions/districts (moyenne)")
    comp = dfv[dfv["maladie"] == choix_maladie]
    comp = comp[comp["annee"].isin(choix_annees)]
    comp = comp.groupby("regions_districts", dropna=True)["incidence_population_pct"].mean().reset_index()
    comp = comp.sort_values("incidence_population_pct", ascending=False).head(20)
    fig2 = px.bar(
        comp, x="regions_districts", y="incidence_population_pct",
        labels={"regions_districts": "Région/District", "incidence_population_pct": "Incidence (%)"},
        title=f"Top régions — {choix_maladie} (moyenne {min(choix_annees)}–{max(choix_annees)})"
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Répartition par année (moyenne)")
    rep = dff.groupby("annee", dropna=True)["incidence_population_pct"].mean().reset_index()
    fig3 = px.pie(rep, names="annee", values="incidence_population_pct", title="Part relative moyenne par année", hole=0.35)
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# 👀 EXPLORATEUR (Pygwalker)
# =========================
with onglets[5]:
    st.header("Explorateur visuel libre (Pygwalker)")
    st.info("Astuce : glissez-déposez les champs à gauche pour créer vos vues interactives.")

    # Test rapide : s'assurer que les composants HTML sont OK
    components.html("<div style='padding:8px;border:1px solid #eee;border-radius:8px'>✅ Test composant HTML OK</div>", height=60)

    if donnees_nettoyees is None or donnees_nettoyees.empty:
        st.warning("Aucune donnée disponible à explorer.")
    else:
        try:
            # Méthode recommandée (API Streamlit de Pygwalker)
            from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
            init_streamlit_comm()
            pyg_html = get_streamlit_html(donnees_nettoyees, use_kernel_calc=True, spec=None)
            components.html(pyg_html, height=950, scrolling=True)
            st.success("Pygwalker (API Streamlit) chargé.")
        except Exception as e_api:
            # Fallback générique (ne bloque pas la page si échec)
            try:
                import pygwalker as pyg
                st.info("Chargement fallback Pygwalker (méthode générique).")
                pyg_html = pyg.to_html(donnees_nettoyees)
                components.html(pyg_html, height=950, scrolling=True)
                st.success("Pygwalker (fallback) chargé.")
            except Exception as e_fallback:
                st.error("Pygwalker n’a pas pu être rendu, mais le reste de l’application reste disponible.")
                st.caption(f"Détails : {type(e_api).__name__ if e_api else ''} {e_api or e_fallback}")

# =========================
# 〽️ MODÉLISATION
# =========================
with onglets[6]:
    st.header("Modélisation supervisée (régression)")

    # Paramètres utilisateur
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        type_modele = st.selectbox(
            "Choisir un modèle",
            ["Régression linéaire", "Régression polynomiale", "Forêt aléatoire", "KNN régression", "Réseau de neurones (ANN)"]
        )
    with colp2:
        test_size = st.slider("Taille test (%)", 10, 40, 20, step=5)
    with colp3:
        random_state = st.number_input("Graine aléatoire", value=42, step=1)

    # Données modèle sans trous (éviter crash scikit-learn)
    df_model = donnees_nettoyees[[
        "annee", "regions_districts", "villes_communes", "maladie", "incidence_population_pct"
    ]].dropna()
    if df_model.empty:
        st.error("Impossible d'entraîner : aucune ligne complète (X et y) après suppression des valeurs manquantes.")
        st.stop()

    X = df_model[["annee", "regions_districts", "villes_communes", "maladie"]]
    y = df_model["incidence_population_pct"]

    colonnes_numeriques = ["annee"]
    colonnes_categorielles = ["regions_districts", "villes_communes", "maladie"]

    preprocesseur = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles)
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=int(random_state)
    )

    # Pipeline modèle
    if type_modele == "Régression linéaire":
        modele = LinearRegression()
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])

    elif type_modele == "Régression polynomiale":
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("mod", LinearRegression())
        ])

    elif type_modele == "Forêt aléatoire":
        modele = RandomForestRegressor(n_estimators=300, random_state=int(random_state))
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])

    elif type_modele == "KNN régression":
        modele = KNeighborsRegressor(n_neighbors=7)
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])

    else:  # Réseau de neurones (ANN)
        modele = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                              max_iter=1000, random_state=int(random_state))
        pipeline = Pipeline([("prep", preprocesseur), ("mod", modele)])

    # Entraînement + Évaluation
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)  # ✅ plus d’avertissement squared=False

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("R²", f"{r2:0.3f}")
    colm2.metric("MAE", f"{mae:0.3f}")
    colm3.metric("RMSE", f"{rmse:0.3f}")

    with st.expander("Voir la validation croisée (5 plis)"):
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
        st.write("Scores R² par pli :", [f"{s:0.3f}" for s in scores])
        st.write("R² moyen :", f"{np.mean(scores):0.3f} ± {np.std(scores):0.3f}")

    # Pipeline disponible pour l’onglet Prédiction
    st.session_state["pipeline_modele"] = pipeline
    st.success("Modèle entraîné et stocké pour l’onglet ◻ Prédiction.")

# =========================
# ◻ PRÉDICTION
# =========================
with onglets[7]:
    st.header("Prédire l’incidence (%) selon vos sélections")

    if "pipeline_modele" not in st.session_state:
        st.warning("Veuillez d’abord entraîner un modèle dans l’onglet 〽️ Modélisation.")
    else:
        pipe = st.session_state["pipeline_modele"]

        if "modele_info" in st.session_state:
            st.info(st.session_state["modele_info"])

        colpr1, colpr2, colpr3, colpr4 = st.columns(4)
        with colpr1:
            annee_sel = st.selectbox("Année", sorted(donnees_nettoyees["annee"].dropna().astype(int).unique()))
        with colpr2:
            region_sel = st.selectbox("Région/District", sorted(donnees_nettoyees["regions_districts"].dropna().unique()))
        with colpr3:
            ville_sel = st.selectbox("Ville/Commune", sorted(donnees_nettoyees["villes_communes"].dropna().unique()))
        with colpr4:
            maladie_sel = st.selectbox("Maladie", sorted(donnees_nettoyees["maladie"].dropna().unique()))

        saisie = pd.DataFrame({
            "annee": [int(annee_sel)],
            "regions_districts": [region_sel],
            "villes_communes": [ville_sel],
            "maladie": [maladie_sel]
        })

        if st.button("🔮 Lancer la prédiction"):
            y_hat = pipe.predict(saisie)[0]
            st.success(f"Incidence attendue : **{y_hat:.2f} %**")

            cond = (
                (donnees_nettoyees["annee"] == int(annee_sel)) &
                (donnees_nettoyees["regions_districts"] == region_sel) &
                (donnees_nettoyees["villes_communes"] == ville_sel) &
                (donnees_nettoyees["maladie"] == maladie_sel)
            )
            ref_local = donnees_nettoyees.loc[cond, "incidence_population_pct"].mean()
            if not np.isnan(ref_local):
                st.info(f"Moyenne observée (mêmes filtres, historique) : **{ref_local:.2f} %**")

# =========================
# 🛖 SOURCE
# =========================
with onglets[8]:
    st.header("Origine des données")
    st.markdown("""
    - **Source ouverte** : *Incidences de maladies sur la population de 2012 à 2015*.  
    - **Lien** : https://data.gouv.ci/datasets/incidence-de-maladies-sur-la-population-de-2012-a-2015
    """)
    st.write("Utilisez les autres onglets pour naviguer dans l’analyse, la modélisation et la prédiction.")
