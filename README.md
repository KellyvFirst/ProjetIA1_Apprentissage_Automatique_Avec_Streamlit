# ProjetIA1_Apprentissage_Automatique_Avec_Streamlit

## Description du projet

`ProjetIA1_Apprentissage_Automatique_Avec_Streamlit` est un tableau de bord de modélisation prédictive qui permet à l'utilisateur de choisir entre la classification et la régression. Il utilise la bibliothèque Streamlit pour créer une interface utilisateur interactive.

## Installation

Assurez-vous d'avoir installé les dépendances nécessaires en exécutant :

```bash
pip install -r requirements.txt

## Utilisation
Pour lancer l’application, exécutez la commande suivante :
streamlit run app.py

## Fonctionnalités
Classification : Utilise le jeu de données pima-indians-diabetes.data.csv et permet à l’utilisateur de choisir entre deux modèles : la régression logistique et l’arbre de décision. Affiche ensuite les résultats du modèle choisi, y compris l’exactitude, la précision, le rappel et le score F1, ainsi que la matrice de confusion.
Régression : Utilise le jeu de données housing.csv et permet à l’utilisateur de choisir entre trois modèles : la régression linéaire, la régression Ridge et Lasso. Affiche ensuite les résultats du modèle choisi, y compris l’erreur absolue moyenne (MAE), l’erreur quadratique moyenne (MSE) et le coefficient de détermination (R^2).

## Contribution
Les contributions sont les bienvenues. Pour toute modification majeure, veuillez ouvrir d’abord une issue pour discuter de ce que vous aimeriez changer.

## Licence
MIT
