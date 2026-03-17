# Prédiction de Survie - Titanic | Python · Scikit-learn · ML

##  Contexte
Le naufrage du Titanic en 1912 est l'une des catastrophes maritimes 
les plus connues. Ce projet utilise les données des 891 passagers 
pour construire un modèle de classification capable de prédire 
la probabilité de survie.

##  Objectif
Identifier les facteurs déterminants de survie et construire 
un modèle prédictif performant en suivant une démarche 
progressive et rigoureuse.

## Stack technique
- **Python** : Pandas, NumPy, Matplotlib, Seaborn
- **Scikit-learn** : Régression Logistique, Random Forest, GridSearchCV
- **GitHub** : Documentation et versioning

## Démarche

### 1. Exploration des données (EDA)
- 891 passagers, 12 variables
- Valeurs manquantes identifiées : Age (177), Cabin (687), Embarked (2)
- Taux de survie global : 38%

### 2. Visualisations clés
- Distribution des survivants par sexe → les femmes survivent 3x plus
- Taux de survie par classe → Classe 1 : 63% vs Classe 3 : 24%
- Matrice de corrélation → Sex et Pclass sont les variables les plus corrélées

### 3. Feature Engineering
Création de 3 nouvelles variables :

| Feature | Description |
|---|---|
| `Title` | Titre extrait du nom (Mr, Mrs, Miss, Master, Officer, Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 si le passager voyage seul |

### 4. Modélisation progressive

| Modèle | Accuracy |
|---|---|
| Régression Logistique | 80.4% |
| Random Forest de base | 80.0% |
| Random Forest + GridSearchCV | 82.1% |
| **Random Forest + Feature Engineering** | **82.7%**  |

### 5. Meilleurs hyperparamètres (GridSearchCV)
```python
{
  'max_depth': 10,
  'min_samples_split': 10,
  'n_estimators': 300
}
```

## Résultats clés

### Importance des features
Le **titre du passager** (variable créée) est la feature 
la plus prédictive avec 25% d'importance devant le 
prix du billet (21%) et le sexe (20%).

### Matrice de confusion finale
|  | Prédit Décédé | Prédit Survivant |
|---|---|---|
| **Réel Décédé** | 93 | 12 |
| **Réel Survivant** | 19 | 55 |

## Insights principaux
- Le sexe et le statut socio-économique sont les facteurs 
  les plus déterminants de survie
- Le feature engineering améliore l'accuracy de +2.3%
- Le titre extrait du nom encode à la fois le genre et 
  le statut social variable très puissante

## Pistes d'amélioration
- Tester XGBoost ou LightGBM
- Explorer le deck de la cabine comme feature
- Appliquer une validation croisée plus robuste (StratifiedKFold)

## Structure du projet
```
projet-titanic/
├── notebooks/
│   └── titanic.ipynb
├── data/
│   └── raw/
│       └── titanic.csv
└── README.md
```

## 🔗 Liens
- [LinkedIn](https://www.linkedin.com/in/safariyatouss)
- [GitHub](https://github.com/Safariyatousani)
