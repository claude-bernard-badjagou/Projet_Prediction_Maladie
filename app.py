
# Application Streamlit : Incidence des maladies (Côte d'Ivoire, 2012-2015)

# -------- Imports (tous commentés) --------
import streamlit as st                     # Importons Streamlit pour construire l'application web interactive
import streamlit.components.v1 as components  # Importons le module des composants HTML pour intégrer Pygwalker
import pandas as pd                        # Importons pandas pour charger et manipuler les données tabulaires
import numpy as np                         # Importons numpy pour quelques opérations numériques
import plotly.express as px                # Importons Plotly Express pour créer des visualisations interactives
import plotly.graph_objects as go          # Importons Graph Objects pour des graphiques personnalisés
from io import StringIO                    # Importons StringIO pour générer un CSV téléchargeable en mémoire

# Outils ML
from sklearn.model_selection import train_test_split, cross_val_score  # Découpage train/test + validation croisée
from sklearn.compose import ColumnTransformer                           # Traitement hétérogène (numérique/catégoriel)
from sklearn.preprocessing import OneHotEncoder, StandardScaler         # Encodage catégoriel + standardisation num
from sklearn.pipeline import Pipeline                                   # Chaînage prétraitements + modèle
from sklearn.linear_model import LinearRegression                       # Régression linéaire
from sklearn.preprocessing import PolynomialFeatures                    # Caractéristiques polynomiales (degré 2)
from sklearn.ensemble import RandomForestRegressor                      # Forêt aléatoire régression
from sklearn.neighbors import KNeighborsRegressor                       # KNN régression
from sklearn.neural_network import MLPRegressor                         # Réseau de neurones (MLP léger)
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error  # Métriques (RMSE moderne)

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
/* Séparateurs visuels confortables */
.section { margin-top: 16px; margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)

# --------- Fonctions utilitaires (toutes en français + commentaires) ---------

@st.cache_data(show_spinner=True)
def charger_donnees_csv(chemin: str) -> pd.DataFrame:
    """Chargeons le fichier CSV pour obtenir le jeu de données source (unique point d'entrée des données)."""
    # Chargeons le fichier CSV pour alimenter toutes les pages de l'application
    df = pd.read_csv(chemin)  # Lecture simple du fichier CSV "Incidence.csv" fourni dans les ressources
    return df                  # Renvoyons le DataFrame chargé

def normaliser_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonisons les libellés de colonnes pour un code robuste, quelles que soient les variantes d'intitulés."""
    donnees = df.copy()  # Copions le DataFrame d’entrée pour travailler proprement

    # Table de correspondance des libellés hétérogènes vers des noms normalisés (sans espaces/accents)
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

    # Renommons les colonnes présentes dans le mapping
    colonnes_renommees = {c: mapping[c] for c in donnees.columns if c in mapping}
    donnees = donnees.rename(columns=colonnes_renommees)

    # Pour plus de robustesse, normalisons tout le reste (minuscules + remplaçons espaces par underscore)
    donnees.columns = [c.strip().lower().replace(" ", "_") for c in donnees.columns]
    return donnees

def typer_colonnes(donnees: pd.DataFrame) -> pd.DataFrame:
    """Typage : convertissons année en float64 (évite les soucis Arrow), incidence en float, autres en string."""
    df = donnees.copy()

    # Chargeons la colonne annee en float64 (garde les NaN et reste Arrow-friendly)
    if "annee" in df.columns:
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce")  # float64 + NaN

    # Chargeons l’incidence en float64
    if "incidence_population_pct" in df.columns:
        df["incidence_population_pct"] = pd.to_numeric(df["incidence_population_pct"], errors="coerce")

    # Forçons les colonnes catégorielles en string pour éviter les surprises plus tard
    for col in ["regions_districts", "villes_communes", "maladie"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

def statistiques_rapides(df: pd.DataFrame) -> pd.DataFrame:
    """Produisons un tableau synthétique des statistiques descriptives numériques pour survol rapide."""
    stats = df.select_dtypes(include=[np.number]).describe().T  # n, moyenne, std, min, max, quartiles
    return stats

def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    """Comptons les valeurs manquantes par colonne pour cibler le nettoyage."""
    na = df.isna().sum().sort_values(ascending=False)
    return pd.DataFrame({"colonne": na.index, "manquants": na.values})

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyons : supprimons les doublons, imputons les NA (mode sur catégoriel, médiane sur numérique)."""
    donnees = df.copy()
    donnees = donnees.drop_duplicates()

    cols_num = donnees.select_dtypes(include=[np.number]).columns.tolist()
    cols_cat = [c for c in donnees.columns if c not in cols_num]

    # Imputons les valeurs manquantes numériques par la médiane (robuste aux outliers)
    for c in cols_num:
        if donnees[c].isna().any():
            donnees[c] = donnees[c].fillna(donnees[c].median())

    # Imputons les valeurs manquantes catégorielles par le mode (valeur la plus fréquente)
    for c in cols_cat:
        if donnees[c].isna().any():
            mode = donnees[c].mode(dropna=True)
            if len(mode) > 0:
                donnees[c] = donnees[c].fillna(mode.iloc[0])

    return donnees

def detecter_valeurs_aberrantes(df: pd.DataFrame, z=3.0) -> pd.DataFrame:
    """Détectons les valeurs aberrantes (z-score > seuil) pour les colonnes numériques."""
    dnum = df.select_dtypes(include=[np.number])
    if dnum.empty:
        return df.iloc[0:0]
    zscores = (dnum - dnum.mean()) / dnum.std(ddof=0)
    masque_out = (zscores.abs() > z).any(axis=1)
    return df.loc[masque_out]

def telecharger_csv(df: pd.DataFrame) -> bytes:
    """Préparons un CSV téléchargeable (en mémoire) pour récupérer les données nettoyées."""
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

def rendre_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertissons les types pandas potentiellement problématiques (nullable, objets mixtes)
    vers des types Arrow-compatibles avant affichage Streamlit.
    """
    dfa = df.copy()
    # Remplaçons pd.NA par np.nan
    dfa = dfa.replace({pd.NA: np.nan})
    # Objets mixtes -> string si ambigu
    for col in dfa.columns:
        if dfa[col].dtype == "object":
            types_uniques = set(type(x) for x in dfa[col].dropna().head(1000))
            if len(types_uniques) > 1:
                dfa[col] = dfa[col].astype("string")
    return dfa

# --------- Chargement unique & préparation initiale ---------
# Chargeons le fichier 'Incidence.csv' pour alimenter l'intégralité de l'app, puis normalisons/typons/Nettoyons
donnees_brutes = charger_donnees_csv("Incidence.csv")          # Chargeons le fichier pour obtenir les données d'origine
donnees = normaliser_noms_colonnes(donnees_brutes)             # Normalisons les libellés pour unifier le code
donnees = typer_colonnes(donnees)                              # Typage cohérent (float64 + string)
donnees_nettoyees = nettoyer_donnees(donnees)                  # Appliquons un nettoyage simple et robuste

# Stockons dans la session pour réutiliser partout sans rechargement
if "donnees_nettoyees" not in st.session_state:
    st.session_state["donnees_nettoyees"] = donnees_nettoyees

# (Option) Auto-entraînement au démarrage pour que l’onglet Prédiction soit utilisable immédiatement
AUTO_TRAIN_ON_START = True  # Mets False si tu préfères entraîner manuellement dans l’onglet Modélisation

def construire_pipeline_defaut():
    """Construisons un pipeline par défaut (prétraitements + RandomForest) pour l'entraînement au démarrage."""
    colonnes_numeriques = ["annee"]
    colonnes_categorielles = ["regions_districts", "villes_communes", "maladie"]
    preprocesseur = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
        ]
    )
    modele = RandomForestRegressor(n_estimators=300, random_state=42)
    pipe = Pipeline([("prep", preprocesseur), ("mod", modele)])
    return pipe

def entrainer_au_demarrage_si_absent(df):
    """Entraînons un pipeline de base et stockons-le en session si aucun modèle n'existe encore."""
    X = df[["annee", "regions_districts", "villes_communes", "maladie"]]
    y = df["incidence_population_pct"]
    pipe = construire_pipeline_defaut()
    pipe.fit(X, y)  # Entraînons le pipeline sur toutes les données pour disposer d'un modèle initial
    st.session_state["pipeline_modele"] = pipe
    st.session_state["modele_info"] = "Modèle par défaut (RandomForest) entraîné au démarrage."

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

    # Petit test pour vérifier que les composants HTML fonctionnent
    components.html("<div style='padding:8px;border:1px solid #eee;border-radius:8px'>✅ Test composant HTML OK</div>", height=60)

    try:
        # Méthode recommandée : API Streamlit de Pygwalker
        from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
        init_streamlit_comm()  # Initialise la communication Streamlit <-> Pygwalker
        pyg_html = get_streamlit_html(donnees_nettoyees, use_kernel_calc=True, spec=None)
        components.html(pyg_html, height=950, scrolling=True)
        st.success("Pygwalker (API Streamlit) chargé.")
    except Exception as e_api:
        # Fallback : méthode générique HTML
        try:
            import pygwalker as pyg
            st.info("Chargement fallback Pygwalker (méthode générique).")
            # Tentons différentes valeurs d’environnement si nécessaire
            pyg_html = None
            for env in ("streamlit", "Jupyter", None):
                try:
                    pyg_html = pyg.to_html(donnees_nettoyees, env=env) if env else pyg.to_html(donnees_nettoyees)
                    if pyg_html and ("<iframe" in pyg_html or "<div" in pyg_html):
                        break
                except Exception:
                    pass
            if not pyg_html:
                raise RuntimeError("Impossible de générer le HTML Pygwalker via la méthode générique.")
            components.html(pyg_html, height=950, scrolling=True)
            st.success("Pygwalker (fallback) chargé.")
        except Exception as e_fallback:
            st.error("Pygwalker n’a pas pu être rendu. Détails ci-dessous :")
            st.exception(e_api if e_api else e_fallback)
            st.caption("Vérifie `pygwalker` et `streamlit` dans requirements.txt.")

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

    # Préparation X / y
    X = donnees_nettoyees[["annee", "regions_districts", "villes_communes", "maladie"]]
    y = donnees_nettoyees["incidence_population_pct"]

    colonnes_numeriques = ["annee"]
    colonnes_categorielles = ["regions_districts", "villes_communes", "maladie"]

    preprocesseur = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=int(random_state)
    )

    # Sélection du pipeline modèle
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
    rmse = root_mean_squared_error(y_test, y_pred)  # ✅ plus de paramètre squared=False

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
