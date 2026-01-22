# Детальный отчёт по кейсам (en)

## Кейсы
### code_001_describe_en (code_generation, language: en, difficulty: easy)

**User input:** Write code that outputs statistics for the dataset.


**Ground truth:**
- expected_code:
```python
df.describe()
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
df.describe(include='all')
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_002_pairplot_en (code_generation, language: en, difficulty: medium)

**User input:** Create a pairplot for all features.


**Ground truth:**
- expected_code:
```python
import plotly.express as px
fig = px.scatter_matrix(df, dimensions=df.columns, color="species")
fig.show()
```
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_002_pairplot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
fig = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
fig.update_traces(diagonal_visible=False)
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_002_pairplot_en_model.png)
- decision_score: True

### code_003_train_model_en (code_generation, language: en, difficulty: medium)

**User input:** Write code that trains a simple classifier on this data.


**Ground truth:**
- expected_code:
```python
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=["species"])
y = df["species"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)
```
- possible_code_objects: RandomForestClassifier, DecisionTreeClassifier, SVC, KNeighborsClassifier, GaussianNB
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_004_missing_check_en (code_generation, language: en, difficulty: easy)

**User input:** Check if there are missing values in the dataset.


**Ground truth:**
- expected_code:
```python
df.isnull().sum()
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
df.isnull().sum()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_005_fill_missing_en (code_generation, language: en, difficulty: easy)

**User input:** Fill missing values with the mean value of column.


**Ground truth:**
- expected_code:
```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_006_boxplot_en (code_generation, language: en, difficulty: medium)

**User input:** Create boxplots for all numeric features.


**Ground truth:**
- expected_code:
```python
import plotly.express as px

fig = px.box(df, y=df.select_dtypes(include="number").columns)
fig.show()
```
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_006_boxplot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

fig = px.box(df, x='species', y='sepal_length', title='Boxplot of Sepal Length by Species')
fig.show()

fig = px.box(df, x='species', y='sepal_width', title='Boxplot of Sepal Width by Species')
fig.show()

fig = px.box(df, x='species', y='petal_length', title='Boxplot of Petal Length by Species')
fig.show()

fig = px.box(df, x='species', y='petal_width', title='Boxplot of Petal Width by Species')
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_006_boxplot_en_model.png)
- decision_score: True

### code_007_standardize_en (code_generation, language: en, difficulty: medium)

**User input:** Standardize all feature columns.


**Ground truth:**
- expected_code:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=["species"]))
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_008_normalize_en (code_generation, language: en, difficulty: medium)

**User input:** Normalize the features in range 0 to 1.


**Ground truth:**
- expected_code:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(df.drop(columns=["species"]))
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_009_pca_en (code_generation, language: en, difficulty: medium)

**User input:** Reduce dimensionality to two components using PCA.


**Ground truth:**
- expected_code:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop(columns=["species"]))
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
from sklearn.decomposition import PCA
import pandas as pd

pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]), columns=['PC1', 'PC2'])
df_pca['species'] = df['species']
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_010_pca_plot_en (code_generation, language: en, difficulty: hard)

**User input:** Make scatter plot of first two PCA components colored by species.


**Ground truth:**
- expected_code:
```python
import plotly.express as px
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop(columns=["species"]))

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["species"] = df["species"]

fig = px.scatter(df_pca, x="PC1", y="PC2", color="species")
fig.show()
```
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_010_pca_plot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# Standardize the features
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df_scaled = (df[features] - df[features].mean()) / df[features].std()

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['species'] = df['species']

# Create scatter plot
fig = px.scatter(pca_df, x='PC1', y='PC2', color='species', title='Scatter Plot of First Two PCA Components Colored by Species')
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_010_pca_plot_en_model.png)
- decision_score: True

### code_011_kmeans_en (code_generation, language: en, difficulty: medium)

**User input:** Make three clusters using KMeans.


**Ground truth:**
- expected_code:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df.drop(columns=["species"]))
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
from sklearn.cluster import KMeans
import pandas as pd

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_012_cluster_plot_en (code_generation, language: en, difficulty: medium)

**User input:** Make scatter plot of KMeans clusters using the first two features.


**Ground truth:**
- expected_code:
```python
import plotly.express as px
from sklearn.cluster import KMeans

X = df.drop(columns=["species"])
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

fig = px.scatter(
    df,
    x="sepal_length",
    y="sepal_width",
    color=clusters.astype(str)
)
fig
```
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_012_cluster_plot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['sepal_length', 'sepal_width']])

px.scatter(df, x='sepal_length', y='sepal_width', color='cluster', symbol='species', title='KMeans Clusters using Sepal Length and Width')
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_012_cluster_plot_en_model.png)
- decision_score: True

### code_013_train_test_split_en (code_generation, language: en, difficulty: easy)

**User input:** Split the dataset into train and test sets.


**Ground truth:**
- expected_code:
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- expected_result:

```python
None
```
- expected_output: Графический вывод
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
from sklearn.model_selection import train_test_split

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_014_show_all_plots_en (code_generation, language: en, difficulty: hard)

**User input:** Show all plots.


**Ground truth:**
- expected_code:
```python
import plotly.express as px
import plotly.graph_objects as go

# df already exists and represents the iris dataset

# Select numeric columns
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Try to infer target column
candidate_targets = ["species", "target", "label", "class"]
target_col = next((c for c in candidate_targets if c in df.columns), None)

if not num_cols:
    raise ValueError("No numeric columns found in df to build plots.")

# 1) Histograms for all numeric features
fig_hist = px.histogram(
    df,
    x=num_cols,
    color=target_col if target_col else None,
    barmode="overlay",
    title="Distributions (Histograms)"
)
fig_hist.show()

# 2) Box plots by class (if target exists)
if target_col:
    for col in num_cols:
        fig_box = px.box(
            df,
            x=target_col,
            y=col,
            title=f"Boxplot: {col} by {target_col}"
        )
        fig_box.show()

# 3) Scatter plots for common Iris feature pairs
scatter_pairs = [
    ("sepal_length", "sepal_width"),
    ("petal_length", "petal_width")
]

for x, y in scatter_pairs:
    if x in df.columns and y in df.columns:
        fig_scatter = px.scatter(
            df,
            x=x,
            y=y,
            color=target_col if target_col else None,
            title=f"Scatter: {x} vs {y}"
        )
        fig_scatter.show()

# 4) Scatter matrix (pairwise relationships)
fig_matrix = px.scatter_matrix(
    df,
    dimensions=num_cols,
    color=target_col if target_col else None,
    title="Scatter Matrix"
)
fig_matrix.show()

# 5) Correlation heatmap
corr = df[num_cols].corr(numeric_only=True)

fig_heatmap = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap"
)
fig_heatmap.show()

```
- possible_code_objects: df, pd, np, px
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_014_show_all_plots_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

fig1 = px.scatter(df, x='sepal_length', y='sepal_width', color='species', title='Sepal Length vs Sepal Width')
fig2 = px.scatter(df, x='petal_length', y='petal_width', color='species', title='Petal Length vs Petal Width')
fig3 = px.box(df, x='species', y='sepal_length', title='Sepal Length by Species')
fig4 = px.box(df, x='species', y='sepal_width', title='Sepal Width by Species')
fig5 = px.box(df, x='species', y='petal_length', title='Petal Length by Species')
fig6 = px.box(df, x='species', y='petal_width', title='Petal Width by Species')

fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
fig6.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_014_show_all_plots_en_model.png)
- decision_score: True

### code_015_histogram_en (code_generation, language: en, difficulty: hard)

**User input:** Show a histogram.


**Ground truth:**
- expected_code:
```python
import plotly.express as px

# df already exists and represents the iris dataset

num_cols = df.select_dtypes(include=["number"]).columns.tolist()

if not num_cols:
    raise ValueError("В df не найдено числовых колонок для построения histogram.")

# Строим histogram по всем числовым признакам
fig = px.histogram(
    df,
    x=num_cols,
    barmode="overlay",
    title="Histogram"
)

fig.show()

```
- possible_code_objects: df, px
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_015_histogram_en_expected.png)


**Model output:**
> Here's the generated code:
```python
fig = px.histogram(df, x='sepal_length', nbins=20, title='Histogram of Sepal Length')
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- decision_score: True

### code_016_index_to_value_en (code_generation, language: en, difficulty: hard)

**User input:** Display the Index to Values plot


**Ground truth:**
- expected_code:
```python
import plotly.express as px

df_reset = df.reset_index().rename(columns={"index": "index"})
fig = px.scatter(
    df_reset,
    x="index",
    y=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
fig
```
- possible_code_objects: df, px
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_016_index_to_value_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
fig = px.line(df, x=df.index, y=df.columns, title='Index to Values Plot')
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot_error: Plotly Express cannot process wide-form data with columns of different type.
- decision_score: True

### code_017_corr_matrix_en (code_generation, language: en, difficulty: hard)

**User input:** Display the correlation matrix plot


**Ground truth:**
- expected_code:
```python
import plotly.express as px

corr = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].corr()
fig = px.imshow(corr, text_auto=True)
fig
```
- possible_code_objects: df, px
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_017_corr_matrix_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
import numpy as np

corr_matrix = df.select_dtypes(include=[np.number]).corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_017_corr_matrix_en_model.png)
- decision_score: True

### code_018_radar_en (code_generation, language: en, difficulty: hard)

**User input:** Display the Radar chart


**Ground truth:**
- expected_code:
```python
import plotly.express as px

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df_mean = df.groupby("species", observed=True)[cols].mean().reset_index()
df_long = df_mean.melt(id_vars="species", var_name="variable", value_name="value")
fig = px.line_polar(df_long, r="value", theta="variable", color="species", line_close=True)
fig
```
- possible_code_objects: df, px
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
None
```
- expected_plot:

 ![expected plot](plots/code_018_radar_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

fig = px.line_polar(df, r=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], theta=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species', line_close=True)
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
None
```
- model_plot_error: All arguments should have the same length. The length of column argument `df[color]` is 150, whereas the length of previously-processed arguments ['r', 'theta'] is 4
- decision_score: True

### chat_001_what_is_iris_en (chat_response, language: en, difficulty: easy)

**User input:** Explain what this dataset represents.


**Ground truth:**
- expected_facts: dataset contains measurements of petals and sepals; three iris species


**Model output:**
> This dataset represents measurements of iris flowers. It includes four features measured in centimeters:  - **sepal_length**: the length of the sepal   - **sepal_width**: the width of the sepal   - **petal_length**: the length of the petal   - **petal_width**: the width of the petal    Each flower also has a **species** label, which is one of three types: setosa, versicolor, or virginica.  These measurements are used to study and classify iris flowers based on their physical characteristics.
- decision_score: True

### chat_002_how_many_classes_en (chat_response, language: en, difficulty: easy)

**User input:** How many classes are in this dataset?


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> There are three classes in this dataset: setosa, versicolor, and virginica. These are the unique values for the 'species' column.
- decision_score: True

### chat_003_best_features_en (chat_response, language: en, difficulty: medium)

**User input:** Which features best separate the iris species?


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> The features that best separate the iris species are **petal_length** and **petal_width**.  These two features show the clearest differences across the three species: setosa, versicolor, and virginica. For example, setosa typically has much smaller petal dimensions compared to the other two species, which helps distinguish it clearly. Versicolor and virginica are more similar, but petal_length and petal_width still provide strong separation between them.  In contrast, sepal_length and sepal_width show more overlap between species, making them less effective for separation.  So, **petal_length** and **petal_width** are the most useful features for distinguishing the iris species.
- decision_score: True

### chat_004_easiest_class_en (chat_response, language: en, difficulty: medium)

**User input:** Which iris species is the easiest to separate from the others?


**Ground truth:**
- expected_facts: setosa; shorter petals; wider petals


**Model output:**
> Based on the dataset features, **setosa** is the easiest iris species to separate from the others.  This is because setosa has distinctly smaller petal and sepal measurements compared to versicolor and virginica. For example, the petal length and petal width for setosa are consistently lower, with little to no overlap with the other two species.  In contrast, versicolor and virginica have overlapping ranges in most features, making them harder to distinguish from each other.  So, **setosa** stands out clearly due to its unique and compact range across all four features.
- decision_score: True

### theory_001_overfitting_en (chat_response, language: en, difficulty: easy)

**User input:** What is overfitting in machine learning?


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> Overfitting happens when a machine learning model learns the training data too well—so well that it starts memorizing the noise and random fluctuations instead of finding the true patterns.  This means the model performs very well on the training data, but poorly on new, unseen data.  For example, if you're predicting species from sepal and petal measurements, an overfitted model might rely too much on tiny, random variations in the training set—like a specific sepal_length value that only appears once—instead of general rules like "setosa tends to have smaller petals."  Overfitting often occurs when a model is too complex, like having too many parameters relative to the amount of training data.  To avoid overfitting, you can use techniques like cross-validation, regularization, or simplifying the model.
- decision_score: True

### theory_002_train_test_split_en (chat_response, language: en, difficulty: easy)

**User input:** Why is a train-test split used?


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> A train-test split is used to evaluate how well a machine learning model will perform on new, unseen data.  By dividing the dataset into two parts — a training set and a test set — we can train the model on the training data and then test it on the test data, which the model has never seen before.  This helps us check if the model is learning general patterns or just memorizing the training data.  For example, if we use all the data to train, we won’t know how well the model works on new examples. With a train-test split, we get a more honest estimate of real-world performance.  In the context of the dataset with **sepal_length**, **sepal_width**, **petal_length**, **petal_width**, and **species**, we might split the data so that the model learns to predict **species** from the measurements, then test how accurately it predicts **species** on the test portion.
- decision_score: True

### theory_003_regularization_en (chat_response, language: en, difficulty: medium)

**User input:** What is regularization and why is it used?


**Ground truth:**
- expected_facts: adds penalty to model complexity; reduces overfitting
- forbidden_facts: regularization increases model variance


**Model output:**
> Regularization is a technique used in machine learning to prevent overfitting. Overfitting happens when a model learns the training data too well, including noise and outliers, so it performs poorly on new, unseen data.  In simple terms, regularization adds a penalty to the model’s complexity. This penalty discourages the model from assigning too much importance to any single feature, which helps it generalize better.  For example, in a model that predicts species based on sepal and petal measurements, regularization might reduce the influence of very large coefficients for features like petal_length or sepal_width. This keeps the model simpler and more reliable when making predictions on new flowers.  Common types include L1 (Lasso) and L2 (Ridge) regularization, which add different kinds of penalties to the loss function during training.  Regularization is especially useful when you have many features or when the dataset is small.
- decision_score: True

### theory_004_bias_variance_en (chat_response, language: en, difficulty: medium)

**User input:** Explain the bias-variance tradeoff.


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> The bias-variance tradeoff is a key concept in machine learning that helps us understand how well a model will generalize to new data.  **Bias** is the error from incorrect assumptions in the learning algorithm. High bias means the model is too simple and doesn’t capture the underlying patterns in the data. For example, using a straight line to fit data that follows a curve leads to high bias.  **Variance** is the error from sensitivity to small fluctuations in the training set. High variance means the model is too complex and learns noise as if it were a pattern. A model that fits every training point perfectly, including outliers, likely has high variance.  The tradeoff happens because as we reduce bias (by making the model more complex), variance tends to increase. Conversely, simplifying the model reduces variance but increases bias.  The goal is to find a balance where both bias and variance are low, so the model performs well on both training and new, unseen data.  For instance, in predicting species from sepal and petal measurements, a very simple model might ignore petal width entirely (high bias), while a very complex model might overfit to small quirks in the training data (high variance). The best model finds a middle ground.
- decision_score: True

### chat_005_feature_count_en (chat_response, language: en, difficulty: easy)

**User input:** How many features does this dataset have?


**Ground truth:**
- expected_facts: four features; sepal length; sepal width; petal length; petal width
- forbidden_facts: five features; only petal features


**Model output:**
> This dataset has four features: sepal_length, sepal_width, petal_length, and petal_width. All are numeric measurements. The fifth column, species, is the target variable, not a feature.
- decision_score: True

### chat_006_samples_count_en (chat_response, language: en, difficulty: easy)

**User input:** How many samples are in this dataset?


**Ground truth:**
- expected_facts: 150 samples; 50 samples per class
- forbidden_facts: 100 samples; imbalanced classes


**Model output:**
> The dataset contains 150 samples.
- decision_score: True

### chat_007_numeric_only_en (chat_response, language: en, difficulty: easy)

**User input:** Are all features in this dataset numeric?


**Ground truth:**
- expected_facts: all features are numeric; continuous measurements
- forbidden_facts: categorical features; text features


**Model output:**
> No, not all features in this dataset are numeric.  The features **sepal_length**, **sepal_width**, **petal_length**, and **petal_width** are all numeric (float type).  However, the **species** feature is a category (category type), which means it contains non-numeric, labeled data like 'setosa', 'versicolor', or 'virginica'.  So, while four of the five features are numeric, one feature—species—is categorical.
- decision_score: True

### chat_008_need_scaling_en (chat_response, language: en, difficulty: medium)

**User input:** Is feature scaling important for this dataset?


**Ground truth:**
- expected_facts: important for distance-based models; helps algorithms like kNN or SVM
- forbidden_facts: scaling is never useful; scaling changes class labels


**Model output:**
> Feature scaling can be important for this dataset, especially if you're using machine learning algorithms that are sensitive to the scale of input features—like k-nearest neighbors, support vector machines, or neural networks.  In this dataset, the features are: - **sepal_length** (float) - **sepal_width** (float) - **petal_length** (float) - **petal_width** (float)  These measurements are in similar units (centimeters), and their ranges are somewhat close. For example, petal length and width tend to be smaller than sepal length and width. But because the scales are not vastly different, scaling might not be critical for all models.  However, if you're using distance-based methods, scaling ensures that no single feature dominates the distance calculation due to its larger magnitude.  So, while scaling isn't always required, it's often a good practice—especially when comparing models or using algorithms sensitive to feature scale.  If you're using a model like logistic regression or a decision tree, scaling may have little effect.  To decide, you could compare model performance with and without scaling. If you'd like, I can help you generate code to check that.
- decision_score: True

### chat_009_linear_separable_en (chat_response, language: en, difficulty: medium)

**User input:** Is this dataset linearly separable?


**Ground truth:**
- expected_facts: setosa is linearly separable; other classes overlap
- forbidden_facts: all classes are perfectly separable; no class is separable


**Model output:**
> I don’t see this in the dataset metadata, so I can’t be sure.  To determine if the dataset is linearly separable, we would need to examine how the species classes are distributed in the feature space — specifically, whether a straight line (or plane in higher dimensions) can separate all species from each other.  Since the dataset includes four features — sepal_length, sepal_width, petal_length, petal_width — and three species, we’d need to visualize or analyze the data to check for linear separability.  Would you like to generate code to visualize the data or check separability using a simple model?
- decision_score: True

### chat_011_unsupervised_possible_en (chat_response, language: en, difficulty: medium)

**User input:** Can this dataset be used for unsupervised learning?


**Ground truth:**
- expected_facts: can be used for clustering; labels can be ignored
- forbidden_facts: unsupervised learning requires labels


**Model output:**
> Yes, this dataset can be used for unsupervised learning.  The features — **sepal_length**, **sepal_width**, **petal_length**, and **petal_width** — are all numerical and suitable for clustering or dimensionality reduction techniques.  For example, you could use algorithms like K-Means or Hierarchical Clustering to find natural groupings in the data without using the **species** label.  However, since the **species** column is available, it’s important to note that unsupervised learning would ignore this label during training. Later, you could compare the discovered clusters to the actual species to evaluate how well the algorithm captured the underlying structure.  So, yes — this dataset supports unsupervised learning tasks using only the numerical features.
- decision_score: True

### chat_012_pca_purpose_en (chat_response, language: en, difficulty: medium)

**User input:** Is PCA applicable to this dataset?


**Ground truth:**
- expected_facts: reduce dimensionality; enable visualization; capture most variance
- forbidden_facts: PCA improves class labels


**Model output:**
> Yes, PCA is applicable to this dataset.  The dataset contains numerical features: sepal_length, sepal_width, petal_length, and petal_width — all of which are continuous and measured in the same units (centimeters). These features are suitable for PCA, which is used to reduce dimensionality by identifying patterns in the data based on variance.  Since the features are on a similar scale and represent measurements of flower parts, PCA can help reveal underlying structure, such as which combinations of features explain most of the variation in the data.  Also, the species column is a categorical label, not used in PCA itself, but it can be used later to interpret the principal components.  So, PCA can be applied to the four numerical features to find the most important directions of variation in the data.
- decision_score: True

### chat_013_outliers_en (chat_response, language: en, difficulty: medium)

**User input:** Are outliers a major concern in this dataset?


**Ground truth:**
- expected_facts: dataset is relatively clean; few or no strong outliers
- forbidden_facts: dataset is heavily corrupted by outliers


**Model output:**
> Outliers can be a concern in any dataset, including this one. In the context of the iris dataset, we look at the four numerical features: sepal_length, sepal_width, petal_length, and petal_width.  These features tend to form tight clusters by species, which suggests that extreme values are rare. For example, petal_length and petal_width are generally small and well-separated between species, making large deviations less likely.  However, without examining the actual data distribution—such as through visualizations like boxplots or statistical measures like interquartile range—we can’t definitively say whether outliers exist.  If you're concerned about outliers, we could check for values that fall significantly outside the expected range for each feature within each species.  Would you like to generate code to detect potential outliers using statistical methods?
- decision_score: True

### chat_014_real_world_limitations_en (chat_response, language: en, difficulty: hard)

**User input:** What are the limitations of this dataset?


**Ground truth:**
- expected_facts: small size; not representative of complex real-world data
- forbidden_facts: suitable for deep learning at scale


**Model output:**
> This dataset has several limitations to keep in mind:  - It only includes four features: sepal_length, sepal_width, petal_length, and petal_width — all measurements in centimeters. There are no other attributes like color, texture, or environmental conditions.  - The target variable is species, with only three categories: setosa, versicolor, and virginica. This limits the dataset’s use for studying more complex classification problems with many classes.  - The dataset is small, with only 150 samples (50 per species). This makes it ideal for learning but less suitable for training complex models that need large amounts of data.  - All data points are from the same source and time period — no variation in collection methods or time-based trends. This reduces its representativeness for real-world scenarios.  - The features are highly correlated, especially petal_length and petal_width. This can affect model performance if not handled properly.  - There are no missing values, but real-world data often has gaps. So, this dataset may not reflect challenges like data imputation.  - It’s a classic educational dataset, so it’s well-separated and easy to classify. Real-world data is often noisier and less cleanly grouped.  If you're using this for a project, consider whether these simplifications affect your goals.
- decision_score: True

### theory_006_classification_vs_regression_en (chat_response, language: en, difficulty: easy)

**User input:** What is the difference between classification and regression?


**Ground truth:**
- expected_facts: classification predicts discrete labels; regression predicts continuous values
- forbidden_facts: classification predicts numbers only; regression predicts categories


**Model output:**
> Classification and regression are both types of supervised machine learning tasks, but they predict different kinds of outcomes.  In **classification**, the goal is to predict a **category** or **label**. For example, in the dataset we have, predicting the species of an iris flower—like setosa, versicolor, or virginica—is a classification task. The output is a discrete class.  In **regression**, the goal is to predict a **continuous number**. For example, predicting the sepal length of an iris flower based on other features would be a regression task. The output is a real number, like 5.1 or 6.3.  So, the main difference is: - **Classification** predicts **which group** an item belongs to. - **Regression** predicts **how much** of something there is.  In our dataset, predicting species is classification. Predicting sepal_length, sepal_width, petal_length, or petal_width would be regression.
- decision_score: True

### theory_007_cross_validation_en (chat_response, language: en, difficulty: medium)

**User input:** What is cross-validation?


**Ground truth:**
- expected_facts: splits data into multiple folds; provides more reliable performance estimates; reduces dependence on a single train-test split
- forbidden_facts: cross-validation guarantees perfect performance


**Model output:**
> Cross-validation is a technique used to evaluate how well a machine learning model will generalize to new, unseen data.  It works by splitting the dataset into multiple parts, or "folds." The model is trained on some of these folds and tested on the remaining ones. This process is repeated so that each fold gets a chance to be used as the test set.  For example, in 5-fold cross-validation, the data is divided into five equal parts. The model is trained on four parts and tested on the fifth. This is done five times, each time using a different part as the test set.  The final performance score is the average of the scores from all five runs.  This helps reduce the risk of overfitting and gives a more reliable estimate of model performance.  In the context of the dataset with features like **sepal_length**, **sepal_width**, **petal_length**, **petal_width**, and **species**, cross-validation can be used to test how well a model predicts the **species** based on the other features.
- decision_score: True

### theory_008_feature_engineering_en (chat_response, language: en, difficulty: medium)

**User input:** What is feature engineering?


**Ground truth:**
- expected_facts: process of creating or transforming features; can improve model performance; requires domain knowledge
- forbidden_facts: feature engineering is only data cleaning


**Model output:**
> Feature engineering is the process of using domain knowledge to create new features or modify existing ones from raw data to improve the performance of machine learning models.  In the context of the dataset with columns like **sepal_length**, **sepal_width**, **petal_length**, **petal_width**, and **species**, feature engineering might involve creating a new feature such as the ratio of petal_length to petal_width, or combining sepal_length and sepal_width into a single measure like a sepal area.  These new features can help the model better capture patterns in the data, like distinguishing between species more clearly.  Remember: Only use the features that are already in the dataset. For example, you can’t create a feature like "petal_area" unless you define it using existing columns like petal_length and petal_width.  If you want to try a specific transformation, we can explore how to do it safely within the dataset’s structure.
- decision_score: True

### theory_009_data_leakage_en (chat_response, language: en, difficulty: medium)

**User input:** What is data leakage?


**Ground truth:**
- expected_facts: information from test data leaks into training; leads to overly optimistic evaluation; harms generalization
- forbidden_facts: data leakage improves real-world performance


**Model output:**
> Data leakage happens when information from outside the training data is used to create a model, giving it an unfair advantage during training. This can make the model seem more accurate than it really is.  For example, if you include a feature that wouldn’t be available when making real predictions—like using a future value or a label from the test set—this is data leakage.  In the context of the dataset with columns: **sepal_length**, **sepal_width**, **petal_length**, **petal_width**, and **species**, data leakage would occur if, for instance, you accidentally used the **species** label to help predict itself during training, or if you included a feature that was derived from test data.  Always make sure your model only uses information that would be available at prediction time.
- decision_score: True

### theory_010_precision_recall_en (chat_response, language: en, difficulty: medium)

**User input:** Explain precision and recall.


**Ground truth:**
- expected_facts: precision measures correctness of positive predictions; recall measures coverage of actual positives; tradeoff between precision and recall
- forbidden_facts: precision and recall are always equal


**Model output:**
> Precision and recall are two important metrics used to evaluate how well a classification model performs, especially when dealing with imbalanced data.  **Precision** answers: *Of all the samples the model predicted as positive, how many were actually positive?*   It measures how accurate the positive predictions are.   High precision means few false positives — the model doesn’t wrongly label negative samples as positive.  **Recall** answers: *Of all the actual positive samples, how many did the model correctly identify?*   It measures how well the model finds all the positive cases.   High recall means few false negatives — the model doesn’t miss many actual positives.  Let’s say we’re classifying flowers into species using the iris dataset.   If we focus on one species, like "setosa", then:  - **Precision** tells us: *Out of all the flowers the model said were setosa, what fraction truly were setosa?* - **Recall** tells us: *Out of all the actual setosa flowers, what fraction did the model catch?*  Both metrics are important.   You might want high precision if you don’t want to mislabel a flower.   You might want high recall if you can’t afford to miss any setosa flowers.  They are calculated using: - True positives (TP): correctly predicted positive cases. - False positives (FP): negative cases wrongly predicted as positive. - False negatives (FN): positive cases wrongly predicted as negative.  **Precision** = TP / (TP + FP)   **Recall** = TP / (TP + FN)  You can’t always maximize both at once — improving one might hurt the other.   That’s why we often use the F1 score, which balances precision and recall.
- decision_score: True

### theory_011_accuracy_limitations_en (chat_response, language: en, difficulty: medium)

**User input:** Is accuracy a universal metric?


**Ground truth:**
- expected_facts: problematic with imbalanced datasets; does not reflect class-wise performance
- forbidden_facts: accuracy is always the best metric


**Model output:**
> Accuracy is not a universal metric. It measures the proportion of correct predictions out of all predictions made. However, it can be misleading in certain situations.  For example, if one species is much more common than the others in the dataset, a model might predict only that species and still achieve high accuracy. But this model would fail to identify the less common species correctly.  In the context of the Iris dataset, where all species have similar numbers of samples, accuracy can be useful. But if the data were unbalanced — for instance, if one species had 90% of the samples — accuracy might hide poor performance on the minority species.  So, while accuracy is simple and intuitive, it’s not always the best choice. Other metrics like precision, recall, or F1-score may be more informative, especially when class distribution is uneven.  Always consider the problem context and data distribution when choosing a metric.
- decision_score: True

### chat_015_what_is_in_data_en (chat_response, language: en, difficulty: easy)

**User input:** What is in the data?


**Ground truth:**
- expected_facts: The data represents the Iris dataset; The dataset contains numeric features describing sepal and petal measurements; There is a categorical feature indicating the iris species; Each row corresponds to a single flower
- forbidden_facts: Mentioning specific statistics or numeric values; Claims about model performance or training results; Assumptions about downstream data processing


**Model output:**
> The data contains measurements of iris flowers with five features:  - **sepal_length**: The length of the sepal in centimeters. - **sepal_width**: The width of the sepal in centimeters. - **petal_length**: The length of the petal in centimeters. - **petal_width**: The width of the petal in centimeters. - **species**: The type of iris flower, one of three categories: setosa, versicolor, or virginica.  All measurements are numerical, and the species is a categorical label.
- decision_score: True
