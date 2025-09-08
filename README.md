### 1. **A propos du projet**

Ce projet est une application interactive développée avec **Streamlit** pour analyser l’**incidence des maladies en Côte d’Ivoire de 2012 à 2015**.
Il comprend plusieurs modules :

* **Exploration des données** (taille, valeurs manquantes, valeurs aberrantes, statistiques descriptives).
* **Manipulation des données** (nettoyage, encodage, sauvegarde en Pickle).
* **Visualisation interactive** (évolution du paludisme, bilharziose, diarrhée, conjonctivite, malnutrition, répartition par régions, maladies les plus fréquentes).
* **Modélisation** avec plusieurs algorithmes de machine learning (régression linéaire, polynomiale, Random Forest, KNN, réseaux de neurones artificiels).
* **Prédictions** des incidences futures en fonction des années, régions et maladies choisies.
* **Origine des données** : issues de la plateforme [data.gouv.ci](https://data.gouv.ci/datasets/incidence-de-maladies-sur-la-population-de-2012-a-2015) .

### 2. **Problématique**

Le projet répond au besoin de :

* **Mieux comprendre l’évolution des maladies en Côte d’Ivoire** (2012–2015) par année, par région, et par type de pathologie.
* **Détecter les tendances et anomalies** dans les données de santé publique.
* **Prédire les incidences futures** pour anticiper les besoins en prévention et en soins.
  👉 Problème central : **comment transformer des données brutes de santé en outils interactifs et prédictifs utiles aux décideurs publics et aux chercheurs ?**

### 3. **Objectif**

* Fournir une **application interactive** permettant d’explorer et de visualiser l’incidence des maladies.
* Développer des **modèles prédictifs fiables** (Random Forest, ANN, etc.) pour estimer les risques futurs.
* Mettre à disposition un **outil d’aide à la décision** pour améliorer les politiques de santé publique, en ciblant mieux les maladies prioritaires et les zones les plus touchées.
* Offrir une **expérience utilisateur simple et pédagogique** (graphiques interactifs, filtres par année, région et maladie).
