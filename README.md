<h3 align='center'>Classify Song Genres from Audio Data<br>Rock or rap?</h3><hr>



<!-- <p align='center'>
  <img width=500 height=300 src='https://i.ytimg.com/vi/oPgWYj2smCw/maxresdefault.jpg'>
</p> -->

<h3>1. Préparer notre dataset</h3>
<p>   Au cours des dernières années, les services de streaming avec d'énormes catalogues sont devenus le principal moyen par lequel la plupart des gens écoutent leur musique préférée. Pour cette raison, les services de streaming ont cherché des moyens de catégoriser la musique pour permettre des recommandations personnalisées.

Chargeons les métadonnées sur nos pistes aux côtés des métriques de piste compilées par<b>The Echo Nest</b>.</p>

```python
import pandas as pd

# Read in track metadata with genre labels
tracks = pd.read_csv('./datasets/fma-rock-vs-hiphop.csv')
# Read in track metrics with the features
echonest_metrics = pd.read_json('./datasets/echonest-metrics.json', precise_float=True)
# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = pd.merge(echonest_metrics, tracks[['track_id', 'genre_top']], on='track_id')
# Inspect the resultant dataframe
print(echo_tracks.info())

```
|   Int64Index: 4802 entries, 0 to 4801   |                       |
|:---------------------------------------:|:---------------------:|
|     Data columns (total 10 columns):    |                       |
|               acousticness              | 4802 non-null float64 |
|               danceability              | 4802 non-null float64 |
|                  energy                 | 4802 non-null float64 |
|             instrumentalness            | 4802 non-null float64 |
|                 liveness                | 4802 non-null float64 |
|               speechiness               | 4802 non-null float64 |
|                  tempo                  | 4802 non-null float64 |
|                 track_id                | 4802 non-null int64   |
|                 valence                 | 4802 non-null float64 |
|                genre_top                | 4802 non-null object  |
| dtypes: float64(8), int64(1), object(1) |                       |
|         memory usage: 412.7+ KB         |                       |


<h3>2. Relations par paires entre variables continues</h3>
<p>Nous voulons éviter d'utiliser des variables qui ont de fortes corrélations entre elles - évitant ainsi la redondance des fonctionnalités
Pour savoir s'il existe des fonctionnalités fortement corrélées dans nos données, nous utiliserons des fonctions intégrées dans le<code>pandas</code> package <code>.corr()</code>. </p>

```python

# Create a correlation matrix
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()
```
<p align='center'>
  <img src='datasets/corr.png'>
</p>

<h3>3. Diviser nos données</h3>

```python

from sklearn.model_selection import train_test_split

# Create features
features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values

# Create labels
labels = echo_tracks["genre_top"].values

# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)
```

<h3>4. Normalisation des données de caractéristiques</h3>

<p>Étant donné que nous n'avons pas trouvé de corrélations fortes particulières entre nos caractéristiques, nous pouvons plutôt utiliser une approche commune pour réduire le nombre de caractéristiques appelée analyse en composantes principales (PCA)
Pour éviter les biais, je normalise d'abord les données à l'aide de la méthode  <code>sklearn</code> built-in <code>StandardScaler</code> method</p>

```python
# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale train_features and set the values to a new variable
scaler = StandardScaler()

# Scale train_features and test_features
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

```

<h3>5. Analyse en composantes principales sur nos données à l'échelle</h3>

<p>PCA est maintenant prêt à déterminer de combien nous pouvons réduire la dimensionnalité de nos données. Nous pouvons utiliser des diagrammes d'éboulis et des diagrammes de rapport expliqué cumulatif pour trouver le nombre de composants à utiliser dans des analyses ultérieures.
Lors de l'utilisation de diagrammes d'éboulis, un «coude» (une forte baisse d'un point de données à l'autre) dans le diagramme est généralement utilisé pour décider d'un seuil approprié.</p>
  
```python
# This is just to make plots appear in the notebook
%matplotlib inline

# Import our plotting module, and PCA class
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')

```

<img src='datasets/PCAhist.png'>

<p>Malheureusement, il ne semble pas y avoir de coude clair dans ce diagramme d'éboulis, ce qui signifie qu'il n'est pas simple de trouver le nombre de dimensions intrinsèques à l'aide de cette méthode.</p>

<h3>6. Visualisation plus poussée de l'ACP</h3>
<p>Examinons maintenant le diagramme de la variance expliquée cumulée pour déterminer combien de caractéristiques sont nécessaires pour expliquer, disons, environ 85 % de la variance</p>

```python
# Import numpy
import numpy as np

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.85.
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

# choose the n_components where about 85% of our variance can be explained
n_components = 6

pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)
```
<img src='datasets/linePCA.png'>

<h3>7. Projection sur nos fonctionnalités </h3>

```python
# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components=6, random_state=10)

# Fit and transform the scaled training features using pca
train_pca = pca.fit_transform(scaled_train_features)

# Fit and transform the scaled test features using pca
test_pca = pca.transform(scaled_test_features)
```

<h3>8. Former decision tree pour classer le genre </h3>
<p>Nous pouvons maintenant utiliser la projection PCA de dimension inférieure des données pour classer les chansons en genres. nous utiliserons un algorithme simple connu sous le nom de <b>decision tree</b>.</p>

```python
# Import Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create our decision tree
tree = DecisionTreeClassifier(random_state=10)

# Train our decision tree
tree.fit(train_pca, train_labels)

# Predict the labels for the test data
pred_labels_tree = tree.predict(test_pca)
```

<h3>9. Comparez notre arbre de décision à  logistic regression</h3>
<p>Il y a toujours la possibilité d'autres modèles qui fonctionneront encore mieux ! Parfois, le plus simple est le meilleur, et nous commencerons donc par appliquer <b>logistic regression</b>.</p>

```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Train our logisitic regression
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

# Create the classification report for both models
from sklearn.metrics import classification_report
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)
```

| Decision Tree: |        |          |         |      |  
|----------------|--------|----------|---------|------|
| precision      | recall | f1-score | support |      |
| Hip-Hop        | 0.66   | 0.66     | 0.66    | 229  |
| Rock           | 0.92   | 0.92     | 0.92    | 972  |
| avg / total    | 0.87   | 0.87     | 0.87    | 1201 |

| Logistic Regression: |        |          |         |      |
|----------------------|--------|----------|---------|------|
| precision            | recall | f1-score | support |      |
| Hip-Hop              | 0.75   | 0.57     | 0.65    | 229  |
| Rock                 | 0.90   | 0.95     | 0.93    | 972  |
| avg / total          | 0.87   | 0.88     | 0.87    | 1201 | 

<h3>10. Équilibrer nos données pour plus de performance</h3>

```python
# Subset a balanced proportion of data points
hop_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop']
rock_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Rock']

# subset only the rock songs, and take a sample the same size as there are hip-hop songs
rock_only = rock_only.sample(hop_only.shape[0], random_state=10)

# concatenate the dataframes hop_only and rock_only
rock_hop_bal = pd.concat([rock_only, hop_only])

# The features, labels, and pca projection are created for the balanced dataframe
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 
labels = rock_hop_bal['genre_top']

# Redefine the train and test set with the pca_projection from the balanced data
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, random_state=10)

train_pca = pca.fit_transform(scaler.fit_transform(train_features))
test_pca = pca.transform(scaler.transform(test_features))
```

<h3>11. L'équilibrage de notre ensemble de données améliore-t-il le biais du modèle ?</h3>

```python
# Train our decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Train our logistic regression on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

# compare the models
print("Decision Tree: \n", classification_report(test_labels, pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))

```
<h3>12. Utiliser la validation croisée pour évaluer nos modèles</h3>
<p>Pour avoir une bonne idée de la performance réelle de nos modèles, nous pouvons appliquer ce qu'on appelle <b>cross-validation</b> (CV).

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
tree_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                      ("tree", DecisionTreeClassifier(random_state=10))])
logreg_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                        ("logreg", LogisticRegression(random_state=10))])

# Set up our K-fold cross-validation
kf = KFold(10)

# Train our models using KFold cv
tree_score = cross_val_score(tree_pipe, features, labels, cv=kf)
logit_score = cross_val_score(logreg_pipe, features, labels, cv=kf)

# Print the mean of each array o scores
print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
>>> Decision Tree: 0.7219780219780221 Logistic Regression: 0.773076923076923

```


