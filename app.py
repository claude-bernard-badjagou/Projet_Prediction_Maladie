
# Application Streamlit : Incidence des maladies (Côte d'Ivoire, 2012-2015)

# -------- Imports (tous commentés) --------
import os                                              # Importons os pour tester l'existence des fichiers locaux
from pathlib import Path                                # Importons Path pour manipuler des chemins de façon robuste
import streamlit as st                                  # Importons Streamlit pour construire l'application web
import pandas as pd                                     # Importons pandas pour charger et manipuler les données tabulaires
import numpy as np                                      # Importons numpy pour quelques opérations numériques
import plotly.express as px                             # Importons Plotly Express pour créer des visualisations interactives
from plotly import graph_objects as go                  # Importons Graph Objects pour des graphiques personnalisés (si besoin)
from io import StringIO                                 # Importons StringIO pour générer un CSV téléchargeable en mémoire
from sklearn.model_selection import train_test_split, cross_val_score  # Outils de split et validation croisée
from sklearn.compose import ColumnTransformer           # ColumnTransformer pour traiter colonnes hétérogènes
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures  # Encodage et features poly
from sklearn.pipeline import Pipeline                   # Pipeline pour chaîner prétraitements + modèle
from sklearn.linear_model import LinearRegression       # Modèle de Régression linéaire
from sklearn.ensemble import RandomForestRegressor      # Modèle de Forêt aléatoire (régression)
from sklearn.neighbors import KNeighborsRegressor       # Modèle KNN régression
from sklearn.neural_network import MLPRegressor        # Réseau de neurones léger (sans TF)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Métriques régression
from pandas.api import types as ptypes                  # Importons outils de détection de types pandas

# (Optionnel) Import propre du composant HTML si jamais on en a besoin ponctuellement
from streamlit.components.v1 import html as html_component  # Importons html(...) via components.v1 (recommandé par la doc)

# -------- Préférences pandas (éviter avertissements futurs inutiles) --------
pd.set_option("future.no_silent_downcasting", True)     # Fixons l’option pour éviter les downcasts silencieux dépréciés

# -------- Utilitaires d’assets --------
def chemin_asset(relatif: str) -> Path:
    """Construisons un chemin absolu sûr vers un asset du dépôt."""
    # Utilisons le répertoire courant de l’app (là où se trouve app.py) pour résoudre la ressource
    racine = Path(__file__).parent
    return (racine / relatif).resolve()

def image_page_icon():
    """Chargeons une icône de page si disponible, sinon utilisons un emoji."""
    # Testons d’abord un chemin 'assets/moustique_tigre.jpg' (bonne pratique : ranger les images dans /assets)
    candidats = ["assets/moustique_tigre.jpg", "moustique_tigre.jpg"]
    for c in candidats:
        p = chemin_asset(c)
        if p.exists():
            return str(p)  # Renvoyons le chemin si l’image est présente
    return "🦟"  # Fallback emoji si l’image n’est pas dans le repo

# -------- Configuration globale de la page --------
st.set_page_config(                                     # Configurons la page Streamlit pour un rendu propre
    page_title="Incidence maladies CI (2012-2015)",     # Définissons le titre de l’onglet navigateur
    page_icon=image_page_icon(),                        # Définissons l’icône de page (image si dispo, sinon emoji)
    layout="wide"                                       # Passons en mode large pour mieux exploiter l’écran
)

# -------- Styles CSS légers pour homogénéiser l'UI --------
st.markdown("""
<style>
.block { background: #ffffff; padding: 16px; border-radius: 12px;
         box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
h2, h3 { color: #0D1D2C; }
[data-testid="stDataFrame"] { border: 1px solid #eee; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --------- Fonctions utilitaires (français + pourquoi) ---------

@st.cache_data(show_spinner=True)
def charger_donnees_csv(chemin: str) -> pd.DataFrame:
    """Chargeons le fichier CSV pour alimenter l’application (source unique)."""
    # Chargeons le fichier CSV pour obtenir le jeu de données source
    df = pd.read_csv(chemin)  # Lecture simple du fichier "Incidence.csv"
    return df                  # Renvoyons le DataFrame chargé

def normaliser_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonisons les libellés de colonnes pour un code robuste (variantes orthographiques incluses)."""
    donnees = df.copy()  # Copions le DataFrame d’entrée pour travailler en sécurité

    # Correspondance libellés hétérogènes -> noms normalisés
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
    # Renommons ce qui est présent
    colonnes_renommees = {c: mapping[c] for c in donnees.columns if c in mapping}
    donnees = donnees.rename(columns=colonnes_renommees)

    # Normalisons tout le reste (minuscules + underscore)
    donnees.columns = [c.strip().lower().replace(" ", "_") for c in donnees.columns]
    return donnees

def typer_colonnes(donnees: pd.DataFrame) -> pd.DataFrame:
    """Typage : convertissons année en entier tolérant NA, incidence en float, autres en string."""
    df = donnees.copy()
    if "annee" in df.columns:
        # Convertissons en entier “nullable” au départ, on adaptera ensuite pour Arrow
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")
    if "incidence_population_pct" in df.columns:
        df["incidence_population_pct"] = pd.to_numeric(df["incidence_population_pct"], errors="coerce")
    for col in ["regions_districts", "villes_communes", "maladie"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df

def statistiques_rapides(df: pd.DataFrame) -> pd.DataFrame:
    """Produisons un tableau synthétique des statistiques numériques pour survol rapide."""
    stats = df.select_dtypes(include=[np.number]).describe().T
    return stats

def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    """Comptons les valeurs manquantes par colonne pour cibler le nettoyage."""
    na = df.isna().sum().sort_values(ascending=False)
    return pd.DataFrame({"colonne": na.index, "manquants": na.values})

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyons : supprimons les doublons, imputons NA (médiane sur numérique, mode sur catégoriel)."""
    donnees = df.copy()
    donnees = donnees.drop_duplicates()

    cols_num = donnees.select_dtypes(include=[np.number, "Int64", "Float64"]).columns.tolist()
    cols_cat = [c for c in donnees.columns if c not in cols_num]

    for c in cols_num:
        if donnees[c].isna().any():
            donnees[c] = donnees[c].fillna(donnees[c].median())

    for c in cols_cat:
        if donnees[c].isna().any():
            mode = donnees[c].mode(dropna=True)
            if len(mode) > 0:
                donnees[c] = donnees[c].fillna(mode.iloc[0])

    return donnees

def rendre_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Convertissons les dtypes 'difficiles' (Int64 nullable, object mixte) pour st.dataframe/Arrow."""
    dfa = df.copy()

    for col in dfa.columns:
        dtype = dfa[col].dtype

        # 1) Entiers "nullable" pandas (Int64, Int32, …) -> float64 si présence de NA (préserve NA)
        if ptypes.is_extension_array_dtype(dtype) and str(dtype).startswith("Int"):
            if dfa[col].isna().any():
                dfa[col] = dfa[col].astype("float64")
            else:
                dfa[col] = dfa[col].astype("int64")

        # 2) Objets hétérogènes : tentons numérique, sinon string
        elif dtype == "object":
            # Essayons une conversion numérique globale
            conv = pd.to_numeric(dfa[col], errors="ignore")
            if conv.dtype != "object":
                dfa[col] = conv
            else:
                dfa[col] = dfa[col].astype("string")

    return dfa

def detecter_valeurs_aberrantes(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Détectons les valeurs aberrantes (z-score > seuil) pour les colonnes numériques."""
    dnum = df.select_dtypes(include=[np.number])
    if dnum.empty:
        return df.head(0)
    zscores = (dnum - dnum.mean()) / dnum.std(ddof=0)
    masque_out = (zscores.abs() > z).any(axis=1)
    return df.loc[masque_out]

def telecharger_csv(df: pd.DataFrame) -> bytes:
    """Préparons un CSV téléchargeable (en mémoire) pour récupérer les données nettoyées."""
    tampon = StringIO()
    df.to_csv(tampon, index=False)
    return tampon.getvalue().encode("utf-8")

# --------- Chargement unique & préparation initiale ---------
# Chargeons le fichier 'Incidence.csv' pour alimenter l'app, puis normalisons/typons/Nettoyons
donnees_brutes = charger_donnees_csv("Incidence.csv")            # Chargeons le fichier CSV pour obtenir les données d'origine
donnees = normaliser_noms_colonnes(donnees_brutes)               # Normalisons les libellés pour unifier le code
donnees = typer_colonnes(donnees)                                # Typage cohérent (entiers, float, string)
donnees_nettoyees = nettoyer_donnees(donnees)                    # Appliquons un nettoyage simple et robuste

# Stockons dans la session pour réutiliser partout sans rechargement
if "donnees_nettoyees" not in st.session_state:
    st.session_state["donnees_nettoyees"] = donnees_nettoyees

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
        # Chargeons l'image uniquement si elle existe pour éviter tout crash (logs précédents)
        img_path = image_page_icon()
        if isinstance(img_path, str) and img_path != "🦟" and Path(img_path).exists():
            st.image(img_path, use_column_width=True, caption="Aedes albopictus (moustique tigre)")
        else:
            st.markdown("### 🦟")  # Fallback visuel minimal

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
# 🛠 EXPLORATION
# =========================
with onglets[2]:
    st.header("Exploration des données")

    # Mettons les tables les unes sous les autres, sans colonnes, comme demandé
    st.subheader("Dimensions & dtypes")
    st.write(f"**Nombre de lignes** : {donnees_nettoyees.shape[0]}")
    st.write(f"**Nombre de colonnes** : {donnees_nettoyees.shape[1]}")
    st.write("**Types de données**")
    st.dataframe(
        rendre_arrow_compatible(donnees_nettoyees.dtypes.to_frame("dtype")),
        use_container_width=True
    )

    st.subheader("Valeurs manquantes par colonne")
    st.dataframe(
        rendre_arrow_compatible(valeurs_manquantes(donnees_nettoyees)),
        use_container_width=True
    )

    st.subheader("Statistiques descriptives (numérique)")
    st.dataframe(
        rendre_arrow_compatible(statistiques_rapides(donnees_nettoyees)),
        use_container_width=True
    )

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
        annees_dispo = sorted(dfv["annee"].dropna().astype(int).unique().tolist())
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
            evol, x="annee", y="incidence_population_pct",
            markers=True, labels={"annee": "Année", "incidence_population_pct": "Incidence (%)"},
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
    fig3 = px.pie(rep, names="annee", values="incidence_population_pct",
                  title="Part relative moyenne par année", hole=0.35)
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# 👀 EXPLORATEUR (Pygwalker)
# =========================
with onglets[5]:
    st.header("Explorateur visuel libre (Pygwalker)")
    st.info("Astuce : glissez-déposez les champs à gauche pour créer vos vues interactives.")

    try:
        # Chargeons l'API Streamlit officielle de Pygwalker (recommandée par la doc)
        from pygwalker.api.streamlit import StreamlitRenderer

        @st.cache_resource(show_spinner=False)
        def obtenir_pyg_renderer(df: pd.DataFrame):
            """Construisons un renderer Pygwalker mis en cache pour de meilleures perfs."""
            # Passons le DataFrame déjà nettoyé ; spec_io_mode="rw" pour permettre sauvegarde de config
            return StreamlitRenderer(df, spec_io_mode="rw")

        renderer = obtenir_pyg_renderer(donnees_nettoyees)
        renderer.explorer(height=900, scrolling=True, default_tab="vis")  # Affichons le studio

    except Exception as e:
        st.error("Pygwalker n’a pas pu être chargé (vérifiez les versions).")
        st.exception(e)
        # Fallback minimal : export HTML si dispo plus tard
        # try:
        #     import pygwalker as pyg
        #     html = pyg.to_html(donnees_nettoyees)
        #     html_component(html, height=900, scrolling=True)
        # except Exception:
        #     pass

# =========================
# 〽️ MODÉLISATION
# =========================
with onglets[6]:
    st.header("Modélisation supervisée (régression)")

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        type_modele = st.selectbox(
            "Choisir un modèle",
            ["Régression linéaire", "Régression polynomiale", "Forêt aléatoire",
             "KNN régression", "Réseau de neurones (ANN)"]
        )
    with colp2:
        taille_test = st.slider("Taille test (%)", 10, 40, 20, step=5)
    with colp3:
        graine = st.number_input("Graine aléatoire", value=42, step=1)

    # Préparons les features/cible
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
        X, y, test_size=taille_test/100, random_state=int(graine)
    )

    # Sélection du pipeline en fonction du modèle choisi
    if type_modele == "Régression linéaire":
        pipeline = Pipeline([("prep", preprocesseur), ("mod", LinearRegression())])

    elif type_modele == "Régression polynomiale":
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("mod", LinearRegression())
        ])

    elif type_modele == "Forêt aléatoire":
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("mod", RandomForestRegressor(n_estimators=300, random_state=int(graine)))
        ])

    elif type_modele == "KNN régression":
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("mod", KNeighborsRegressor(n_neighbors=7))
        ])

    else:
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("mod", MLPRegressor(hidden_layer_sizes=(128, 64),
                                 activation="relu", solver="adam",
                                 max_iter=1000, random_state=int(graine)))
        ])

    # Entraînons le modèle
    pipeline.fit(X_train, y_train)

    # Évaluons sur le test
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # conforme sklearn>=1.4

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("R²", f"{r2:0.3f}")
    cm2.metric("MAE", f"{mae:0.3f}")
    cm3.metric("RMSE", f"{rmse:0.3f}")

    with st.expander("Voir la validation croisée (5 plis)"):
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
        st.write("Scores R² par pli :", [f"{s:0.3f}" for s in scores])
        st.write("R² moyen :", f"{np.mean(scores):0.3f} ± {np.std(scores):0.3f}")

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
