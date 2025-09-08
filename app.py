# ================================================================
# Application Streamlit : Incidence des maladies (Côte d'Ivoire, 2012–2015)
# Fichier unique, pages en onglets, variables/fonctions en français,
# commentaires ligne par ligne, prêt pour déploiement Streamlit.
# ================================================================

# ------------- Importations (toutes commentées) -------------
from pathlib import Path                               # Pour gérer les chemins robustement (assets, CSV)
import io                                              # Pour créer des contenus téléchargeables en mémoire
import numpy as np                                     # Pour les calculs numériques (statistiques, z-score)
import pandas as pd                                    # Pour manipuler les données tabulaires
import plotly.express as px                            # Pour des graphiques interactifs rapides
import streamlit as st                                 # Pour construire l’interface web
from sklearn.model_selection import train_test_split, cross_val_score  # Pour split et validation croisée
from sklearn.compose import ColumnTransformer          # Pour appliquer des transformations par type de colonne
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures  # Encodage/scaling/features
from sklearn.pipeline import Pipeline                  # Pour enchaîner prétraitements + modèle
from sklearn.linear_model import LinearRegression      # Modèle de régression linéaire
from sklearn.ensemble import RandomForestRegressor     # Modèle de régression par forêts aléatoires
from sklearn.neighbors import KNeighborsRegressor      # Modèle KNN régression
from sklearn.neural_network import MLPRegressor       # Réseau de neurones (petit MLP scikit-learn)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Métriques de régression
from pandas.api import types as ptypes                 # Pour détecter proprement les dtypes pandas

# ------------- Préférences pandas (éviter warnings verbeux) -------------
pd.set_option("future.no_silent_downcasting", True)    # Déclarons l’option futur pour éviter les downcasts silencieux

# ------------- Résolution des chemins d’assets (image, CSV) -------------
def chemin_depuis_app(relatif: str) -> Path:
    """Retourner un chemin absolu à partir du dossier contenant ce app.py."""
    return (Path(__file__).parent / relatif).resolve()  # Construisons un chemin absolu robuste

def chemin_image_page() -> str | None:
    """Chercher l’icône d’onglet dans assets/, sinon retourner None."""
    p = chemin_depuis_app("assets/moustique_tigre.jpg")  # Chargeons le fichier assets/moustique_tigre.jpg pour l’icône
    return str(p) if p.exists() else None               # Renvoyons le chemin si trouvée, sinon None

# ------------- Configuration Streamlit -------------
st.set_page_config(                                     # Configurons la page pour un rendu large et propre
    page_title="Incidence maladies CI (2012–2015)",     # Titre d’onglet navigateur
    page_icon=(chemin_image_page() or "🦟"),            # Icône: image si dispo, sinon emoji
    layout="wide"                                       # Mise en page pleine largeur
)

# ------------- Style léger (CSS) -------------
st.markdown("""
<style>
.block { background:#fff; padding:16px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
h2, h3 { color:#0D1D2C; }
[data-testid="stDataFrame"] { border:1px solid #eee; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# Fonctions utilitaires (en français, *ce qui est fait et pourquoi*)
# ================================================================

@st.cache_data(show_spinner=True, ttl=3600)
def charger_donnees_csv(chemin_csv: str) -> pd.DataFrame:
    """Chargeons le fichier CSV pour alimenter l’application sans relire le disque à chaque interaction."""
    df = pd.read_csv(chemin_csv)                        # Lisons le CSV fourni (Incidence.csv)
    return df                                           # Renvoyons le DataFrame brut

def normaliser_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonisons les libellés de colonnes pour gérer les variantes et coder de façon stable."""
    donnees = df.copy()                                 # Copions pour éviter les effets de bord
    mapping = {                                         # Définissons un mapping des variantes -> noms standard
        "ANNEE": "annee",
        "REGIONS / DISTRICTS": "regions_districts",
        "REGIONS/DISTRICTS": "regions_districts",
        "VILLES / COMMUNES": "villes_communes",
        "VILLES/COMMUNES": "villes_communes",
        "MALADIE": "maladie",
        "INCIDENCE SUR LA POPULATION GENERALE (%)": "incidence_population_pct",
        "INCIDENCE_SUR_LA_POPULATION_GENERALE_(%)": "incidence_population_pct",
    }
    renoms = {c: mapping[c] for c in donnees.columns if c in mapping}  # Conservons uniquement les clés présentes
    donnees = donnees.rename(columns=renoms)           # Appliquons le renommage présent
    donnees.columns = [c.strip().lower().replace(" ", "_") for c in donnees.columns]  # Uniformisons le reste
    return donnees                                     # Renvoyons un schéma de colonnes stable

def typer_colonnes(donnees: pd.DataFrame) -> pd.DataFrame:
    """Typage contrôlé : année en entier tolérant NA, incidence en float, catégories en string (pandas)."""
    df = donnees.copy()                                # Copions le DF d’entrée
    if "annee" in df.columns:                          # Si la colonne annee existe
        df["annee"] = pd.to_numeric(df["annee"], errors="coerce").astype("Int64")  # Convertissons en entier nullable
    if "incidence_population_pct" in df.columns:       # Si la colonne incidence existe
        df["incidence_population_pct"] = pd.to_numeric(df["incidence_population_pct"], errors="coerce")  # En float
    for c in ["regions_districts", "villes_communes", "maladie"]:  # Pour chaque colonne catégorielle
        if c in df.columns:
            df[c] = df[c].astype("string")             # Forçons un type string propre
    return df                                          # Renvoyons le DF typé

def statistiques_rapides(df: pd.DataFrame) -> pd.DataFrame:
    """Calculons les statistiques descriptives pour les colonnes numériques (survol rapide)."""
    return df.select_dtypes(include=[np.number]).describe().T  # Utilisons describe() puis transposons pour lisibilité

def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    """Comptons les valeurs manquantes par colonne pour cibler le nettoyage."""
    na = df.isna().sum().sort_values(ascending=False)  # Comptons les NA
    return pd.DataFrame({"colonne": na.index, "manquants": na.values})  # Renvoyons un tableau clair

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyons : doublons supprimés, imputations (médiane en numérique, mode en catégoriel) pour fiabiliser l’EDA."""
    donnees = df.copy()                                # Copions
    donnees = donnees.drop_duplicates()                # Supprimons les doublons ligne à ligne

    cols_num = donnees.select_dtypes(include=[np.number, "Int64", "Float64"]).columns.tolist()  # Numériques
    cols_cat = [c for c in donnees.columns if c not in cols_num]  # Catégorielles = complément

    for c in cols_num:                                 # Parcourons les numériques
        if donnees[c].isna().any():                    # Si NA présents
            donnees[c] = donnees[c].fillna(donnees[c].median())  # Remplaçons par la médiane (robuste)

    for c in cols_cat:                                 # Parcourons les catégorielles
        if donnees[c].isna().any():                    # Si NA présents
            mode = donnees[c].mode(dropna=True)        # Calculons la modalité la plus fréquente
            if len(mode) > 0:
                donnees[c] = donnees[c].fillna(mode.iloc[0])  # Remplaçons par ce mode

    return donnees                                     # Renvoyons la version nettoyée

def rendre_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Rendons le DF compatible Arrow/Streamlit: on convertit Int64 nullable et object hétérogène pour éviter les correctifs auto."""
    dfa = df.copy()                                    # Copions le DF
    for col in dfa.columns:                            # Parcourons chaque colonne
        dtype = dfa[col].dtype                         # Récupérons le dtype
        # Entiers "nullable" (Int64, Int32, …) → float64 s’il y a NA (préserve NA), sinon int64 pur
        if ptypes.is_extension_array_dtype(dtype) and str(dtype).startswith("Int"):
            if dfa[col].isna().any():
                dfa[col] = dfa[col].astype("float64")  # Passons en float64 pour gérer NA proprement
            else:
                dfa[col] = dfa[col].astype("int64")    # Sinon int64 natif
        # Objets hétérogènes → tentons numérique, sinon string explicite
        elif dtype == "object":
            conv = pd.to_numeric(dfa[col], errors="ignore")  # Tentons conversion
            dfa[col] = conv if conv.dtype != "object" else dfa[col].astype("string")  # Forçons string si nécessaire
    return dfa                                          # Renvoyons un DF Arrow-safe

def detecter_valeurs_aberrantes(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Détectons des lignes présentant au moins un z-score > z sur les variables numériques (repérage d’anomalies)."""
    dnum = df.select_dtypes(include=[np.number])       # Conservons uniquement le numérique
    if dnum.empty:                                     # Si rien de numérique
        return df.head(0)                              # Renvoyons un DF vide
    zscores = (dnum - dnum.mean()) / dnum.std(ddof=0)  # Calculons les z-scores
    masque_out = (zscores.abs() > z).any(axis=1)       # Marquons les lignes aberrantes
    return df.loc[masque_out]                          # Renvoyons ces lignes

def telecharger_csv(df: pd.DataFrame) -> bytes:
    """Convertissons un DataFrame en CSV (bytes) pour téléchargement via st.download_button sans écrire sur disque."""
    tampon = io.StringIO()                             # Ouvrons un tampon texte en mémoire
    df.to_csv(tampon, index=False)                     # Écrivons le CSV
    return tampon.getvalue().encode("utf-8")           # Retournons les octets UTF-8

# ================================================================
# Chargement & préparation (fait une seule fois grâce au cache)
# ================================================================

# Chargeons le CSV source (nom imposé : Incidence.csv à la racine du dépôt)
donnees_brutes = charger_donnees_csv(chemin_depuis_app("Incidence.csv"))  # Chargeons le jeu brut
donnees = normaliser_noms_colonnes(donnees_brutes)                        # Normalisons les colonnes
donnees = typer_colonnes(donnees)                                         # Appliquons un typage cohérent
donnees_nettoyees = nettoyer_donnees(donnees)                             # Nettoyons le dataset

# Conservons le DF nettoyé en session pour réutiliser sans recalcul
if "donnees_nettoyees" not in st.session_state:                           # Si pas encore présent
    st.session_state["donnees_nettoyees"] = donnees_nettoyees             # Stockons en session

# ================================================================
# Navigation horizontale par onglets
# ================================================================
onglets = st.tabs([
    "🏠 Accueil", "📒 Informations", "🛠 Exploration", "🧹 Préparation",
    "🔍 Visualisations", "👀 Explorateur", "〽️ Modélisation", "◻ Prédiction", "🛖 Source"
])

# =========================
# 🏠 ACCUEIL
# =========================
with onglets[0]:
    st.title("Incidence des maladies en Côte d’Ivoire (2012–2015)")      # Posons un titre clair
    col1, col2 = st.columns([1, 2], gap="large")                          # Deux colonnes pour une intro agréable
    with col1:
        p_img = chemin_depuis_app("assets/moustique_tigre.jpg")           # Chargeons l’image d’illustration depuis assets/
        if p_img.exists():
            st.image(str(p_img), use_column_width=True,
                     caption="Aedes albopictus (moustique tigre)")        # Affichons l’image si disponible
        else:
            st.markdown("### 🦟")                                         # Fallback visuel léger si absente
    with col2:
        st.markdown("""<div class="block" style="text-align:justify">
        <h3>Objectif de l’application</h3>
        Explorer, visualiser et modéliser l’incidence de maladies en Côte d’Ivoire (2012–2015) : 
        graphiques interactifs, explorateur visuel (Pygwalker) et modèles prédictifs (Régression, RF, KNN, MLP).
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="block" style="text-align:justify">
        <h3>Problème adressé</h3>
        Transformer des données brutes de santé publique en <b>indicateurs actionnables</b> et en <b>prédictions</b> 
        pour aider à prioriser les zones et pathologies.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="block" style="text-align:justify">
        <h3>Résultats attendus</h3>
        Un outil unique pour : (1) analyser les tendances, (2) comparer les régions, 
        (3) prédire l’incidence selon année/région/ville/maladie.
        </div>""", unsafe_allow_html=True)

# =========================
# 📒 INFORMATIONS
# =========================
with onglets[1]:
    st.header("Informations sur les données")                             # Annonçons la section
    st.write("**Aperçu des premières lignes (jeu d’origine)**")           # Présentons l’aperçu
    st.dataframe(                                                         # Affichons un échantillon Arrow-safe
        rendre_arrow_compatible(donnees_brutes.head()),
        use_container_width=True
    )
    st.write("**Libellés de colonnes normalisés (utilisés en interne)**") # Expliquons les noms internes
    st.json({
        "annee": "Année d’observation (2012–2015)",
        "regions_districts": "Région ou district sanitaire",
        "villes_communes": "Ville ou commune",
        "maladie": "Type de pathologie",
        "incidence_population_pct": "Incidence sur la population générale (en %)"
    })

# =========================
# 🛠 EXPLORATION (tables empilées verticalement)
# =========================
with onglets[2]:
    st.header("Exploration des données")                                   # Titre de section
    st.subheader("Dimensions & dtypes")                                    # Sous-titre
    st.write(f"**Nombre de lignes** : {donnees_nettoyees.shape[0]}")       # Indiquons le nb de lignes
    st.write(f"**Nombre de colonnes** : {donnees_nettoyees.shape[1]}")     # Indiquons le nb de colonnes
    st.dataframe(                                                          # Dtypes lisibles
        rendre_arrow_compatible(donnees_nettoyees.dtypes.to_frame("dtype")),
        use_container_width=True
    )

    st.subheader("Valeurs manquantes par colonne")                         # Sous-titre
    st.dataframe(
        rendre_arrow_compatible(valeurs_manquantes(donnees_nettoyees)),
        use_container_width=True
    )

    st.subheader("Statistiques descriptives (numériques)")                 # Sous-titre
    st.dataframe(
        rendre_arrow_compatible(statistiques_rapides(donnees_nettoyees)),
        use_container_width=True
    )

    st.subheader("Valeurs aberrantes potentielles (z-score > 3)")          # Sous-titre
    outliers = detecter_valeurs_aberrantes(donnees_nettoyees)              # Repérons les anomalies
    if outliers.empty:                                                     # Si rien d’aberrant
        st.info("Aucune ligne ne dépasse le seuil de z-score sélectionné (3.0).")
    else:
        st.dataframe(rendre_arrow_compatible(outliers), use_container_width=True)

# =========================
# 🧹 PRÉPARATION (export)
# =========================
with onglets[3]:
    st.header("Préparation et export des données")                         # Titre de section
    st.write("**Aperçu après nettoyage (doublons supprimés, NA imputés)**")
    st.dataframe(
        rendre_arrow_compatible(donnees_nettoyees.head(20)),
        use_container_width=True
    )
    st.download_button(                                                    # Proposons un export CSV
        label="📥 Télécharger les données nettoyées (CSV)",
        data=telecharger_csv(donnees_nettoyees),
        file_name="incidence_nettoyee.csv",
        mime="text/csv"
    )

# =========================
# 🔍 VISUALISATIONS (filtres + 3 graphiques)
# =========================
with onglets[4]:
    st.header("Visualisations interactives")                               # Titre de section
    dfv = donnees_nettoyees                                                # Alias local

    colf1, colf2, colf3 = st.columns(3)                                    # 3 filtres côte à côte
    with colf1:
        maladies_dispo = sorted(dfv["maladie"].dropna().unique().tolist())  # Liste des maladies
        choix_maladie = st.selectbox("Choisir une maladie", maladies_dispo, index=0)
    with colf2:
        annees_dispo = sorted(dfv["annee"].dropna().astype(int).unique().tolist())  # Liste des années
        choix_annees = st.multiselect("Filtrer par année", annees_dispo, default=annees_dispo)
    with colf3:
        regions_dispo = ["(Toutes)"] + sorted(dfv["regions_districts"].dropna().unique().tolist())  # Régions
        choix_region = st.selectbox("Filtrer par région/district", regions_dispo, index=0)

    dff = dfv[dfv["maladie"] == choix_maladie]                             # Appliquons filtre maladie
    dff = dff[dff["annee"].isin(choix_annees)]                             # Filtrons les années
    if choix_region != "(Toutes)":                                         # Si région choisie
        dff = dff[dff["regions_districts"] == choix_region]                # Filtrons la région

    st.subheader("Évolution de l’incidence (%) dans le temps")             # Graphique 1
    if len(dff) == 0:
        st.warning("Aucune donnée pour ce filtre.")
    else:
        evol = dff.groupby("annee", dropna=True)["incidence_population_pct"].mean().reset_index()
        fig1 = px.line(evol, x="annee", y="incidence_population_pct",
                       markers=True,
                       labels={"annee": "Année", "incidence_population_pct": "Incidence (%)"},
                       title=f"Évolution — {choix_maladie}")
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Comparaison des régions/districts (moyenne)")            # Graphique 2
    comp = dfv[dfv["maladie"] == choix_maladie]
    comp = comp[comp["annee"].isin(choix_annees)]
    comp = comp.groupby("regions_districts", dropna=True)["incidence_population_pct"].mean().reset_index()
    comp = comp.sort_values("incidence_population_pct", ascending=False).head(20)
    fig2 = px.bar(comp, x="regions_districts", y="incidence_population_pct",
                  labels={"regions_districts": "Région/District", "incidence_population_pct": "Incidence (%)"},
                  title=f"Top régions — {choix_maladie} (moyenne {min(choix_annees)}–{max(choix_annees)})")
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Répartition par année (moyenne)")                        # Graphique 3
    rep = dff.groupby("annee", dropna=True)["incidence_population_pct"].mean().reset_index()
    fig3 = px.pie(rep, names="annee", values="incidence_population_pct", title="Part relative moyenne par année", hole=0.35)
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# 👀 EXPLORATEUR (Pygwalker, stable)
# =========================
with onglets[5]:
    st.header("Explorateur visuel libre (Pygwalker)")                      # Titre de section
    st.info("Astuce : glissez-déposez les champs à gauche pour créer vos vues interactives.")
    try:
        # Importons le renderer officiel Streamlit (plus stable qu’un HTML brut)
        from pygwalker.api.streamlit import StreamlitRenderer             # Importons l’API intégrée

        @st.cache_resource(show_spinner=False)
        def obtenir_renderer(df: pd.DataFrame):
            """Construisons un renderer Pygwalker mis en cache pour éviter des re-instanciations coûteuses."""
            return StreamlitRenderer(df, spec_io_mode="rw")               # Autorisons lecture/écriture de la spec localement

        renderer = obtenir_renderer(donnees_nettoyees)                    # Instancions le renderer (caché)
        renderer.explorer(height=900, scrolling=True, default_tab="vis")  # Ouvrons l’explorateur

    except Exception as e:
        st.error("Pygwalker n’a pas pu être chargé. Vérifiez l’environnement et la version installée.")
        st.exception(e)                                                   # Montrons l’exception pour diagnostic

# =========================
# 〽️ MODÉLISATION (entraînement à la demande)
# =========================
with onglets[6]:
    st.header("Modélisation supervisée (régression)")                     # Titre de section
    colp1, colp2, colp3 = st.columns(3)                                   # Paramètres utilisateur
    with colp1:
        type_modele = st.selectbox(
            "Choisir un modèle",
            ["Régression linéaire", "Régression polynomiale", "Forêt aléatoire",
             "KNN régression", "Réseau de neurones (ANN)"]
        )
    with colp2:
        taille_test = st.slider("Taille test (%)", 10, 40, 20, step=5)    # Taille test
    with colp3:
        graine = st.number_input("Graine aléatoire", value=42, step=1)    # Graine

    # Préparons X/y (caractéristiques et cible)
    X = donnees_nettoyees[["annee", "regions_districts", "villes_communes", "maladie"]]
    y = donnees_nettoyees["incidence_population_pct"]

    colonnes_numeriques = ["annee"]                                       # Numérique
    colonnes_categorielles = ["regions_districts", "villes_communes", "maladie"]  # Catégorielles

    preprocesseur = ColumnTransformer(                                     # Prétraitement par type
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles)
        ]
    )

    # Découpons apprentissage/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=taille_test/100.0, random_state=int(graine)
    )

    # Construisons le pipeline selon le choix
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

    else:  # Réseau de neurones
        pipeline = Pipeline([
            ("prep", preprocesseur),
            ("mod", MLPRegressor(hidden_layer_sizes=(128, 64),
                                 activation="relu", solver="adam",
                                 max_iter=1000, random_state=int(graine)))
        ])

    # Bouton explicite d’entraînement (évite de réentraîner à chaque changement mineur)
    if st.button("🔁 Entraîner / Réentraîner le modèle"):
        pipeline.fit(X_train, y_train)                                     # Entraînons le pipeline
        y_pred = pipeline.predict(X_test)                                  # Prédictions test
        r2 = r2_score(y_test, y_pred)                                      # R²
        mae = mean_absolute_error(y_test, y_pred)                          # MAE
        rmse = mean_squared_error(y_test, y_pred, squared=False)           # RMSE (API récente)

        c1, c2, c3 = st.columns(3)                                         # Montrons les métriques
        c1.metric("R²", f"{r2:.3f}")
        c2.metric("MAE", f"{mae:.3f}")
        c3.metric("RMSE", f"{rmse:.3f}")

        with st.expander("Voir la validation croisée (5 plis)"):
            scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
            st.write("Scores R² par pli :", [f"{s:.3f}" for s in scores])
            st.write("R² moyen :", f"{np.mean(scores):.3f} ± {np.std(scores):.3f}")

        st.session_state["pipeline_modele"] = pipeline                     # Stockons le modèle en session
        st.success("✅ Modèle entraîné et stocké pour l’onglet ◻ Prédiction.")

    else:
        st.info("Cliquez sur **Entraîner / Réentraîner le modèle** pour calculer les métriques et activer la prédiction.")

# =========================
# ◻ PRÉDICTION
# =========================
with onglets[7]:
    st.header("Prédire l’incidence (%) selon vos sélections")             # Titre de section
    if "pipeline_modele" not in st.session_state:                         # Vérifions qu’un modèle existe
        st.warning("Aucun modèle disponible. Allez dans **〽️ Modélisation** puis entraînez le modèle.")
    else:
        pipe = st.session_state["pipeline_modele"]                        # Récupérons le pipeline

        col1, col2, col3, col4 = st.columns(4)                            # Quatre critères d’entrée
        with col1:
            annee_sel = st.selectbox("Année", sorted(donnees_nettoyees["annee"].dropna().astype(int).unique()))
        with col2:
            region_sel = st.selectbox("Région/District", sorted(donnees_nettoyees["regions_districts"].dropna().unique()))
        with col3:
            ville_sel = st.selectbox("Ville/Commune", sorted(donnees_nettoyees["villes_communes"].dropna().unique()))
        with col4:
            maladie_sel = st.selectbox("Maladie", sorted(donnees_nettoyees["maladie"].dropna().unique()))

        # Construisons une observation à une ligne avec les mêmes noms de colonnes que X
        saisie = pd.DataFrame({
            "annee": [int(annee_sel)],
            "regions_districts": [region_sel],
            "villes_communes": [ville_sel],
            "maladie": [maladie_sel]
        })

        if st.button("🔮 Lancer la prédiction"):
            y_hat = float(pipe.predict(saisie)[0])                         # Calculons la prédiction (float)
            st.success(f"Incidence attendue : **{y_hat:.2f} %**")          # Affichons le résultat

            # Calculons en repère la moyenne historique locale si disponible
            cond = (
                (donnees_nettoyees["annee"] == int(annee_sel)) &
                (donnees_nettoyees["regions_districts"] == region_sel) &
                (donnees_nettoyees["villes_communes"] == ville_sel) &
                (donnees_nettoyees["maladie"] == maladie_sel)
            )
            ref_local = donnees_nettoyees.loc[cond, "incidence_population_pct"].mean()
            if not np.isnan(ref_local):
                st.info(f"Moyenne observée (historique, mêmes filtres) : **{ref_local:.2f} %**")

# =========================
# 🛖 SOURCE
# =========================
with onglets[8]:
    st.header("Origine des données")
    st.markdown("""
    - **Source ouverte** : *Incidences de maladies sur la population de 2012 à 2015*  
    - **Lien** : https://data.gouv.ci/datasets/incidence-de-maladies-sur-la-population-de-2012-a-2015
    """)
    st.caption("Placez **Incidence.csv** à la racine et l’image sous **assets/moustique_tigre.jpg**.")
