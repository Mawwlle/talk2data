# Детальный отчёт по кейсам (en)

## Методика расчёта метрик
### Semantic similarity (только для chat_response)
Формула:
```text
semantic_similarity = 0.6 * similarity + 0.4 * expected_coverage - 0.5 * forbidden_penalty
```
Где доли:
 - expected_coverage — доля ожидаемых фактов, упомянутых в ответе;
 - forbidden_penalty — доля запрещённых фактов,
попавших в ответ (штраф).
 - similarity — embedding-cosine между эталонным ответом (конкатенация expected_facts или базовый
референс) и ответом модели.


Факт считается покрытым, если косинусная близость fact↔ответ ≥ 0.55
(или высокая токеновая схожесть), что позволяет засчитывать перефраз. Весами (0.6/0.4/0.5) балансируем близость текста и полноту фактов,
давая штраф за запрещённые факты, чтобы сохранить интерпретируемость (веса суммарно ограничивают метрику в [0, 1]).

### Code score (только для code_generation)
- heuristic_score = 0.3 (валидный синтаксис) + до 0.2 (совпадение импортов) + до 0.4 (совпадение ключевых вызовов).
- result_match: бинарный флаг (1.0/0.0), что итоговый результат выполнения совпал с эталоном без ошибок исполнения.

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
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
stats = df.describe(include='all')
stats
```
- model_result:

```python
        sepal_length  sepal_width  petal_length  petal_width species
count     150.000000   150.000000    150.000000   150.000000     150
unique           NaN          NaN           NaN          NaN       3
top              NaN          NaN           NaN          NaN  setosa
freq             NaN          NaN           NaN          NaN      50
mean        5.843333     3.057333      3.758000     1.199333     NaN
std         0.828066     0.435866      1.765298     0.762238     NaN
min         4.300000     2.000000      1.000000     0.100000     NaN
25%         5.100000     2.800000      1.600000     0.300000     NaN
50%         5.800000     3.000000      4.350000     1.300000     NaN
75%         6.400000     3.300000      5.100000     1.800000     NaN
max         7.900000     4.400000      6.900000     2.500000     NaN
```
- model_error:

```python
None
```


**Метрики:**
- decision_score: True
  - heuristic_score: 1.0
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected:  | model:  | matched:
    - calls_score: 0.444 | expected: describe | model: describe | matched: describe

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
execution_timeout: sandbox_timeout
```
- expected_plot:

 ![expected plot](plots/code_002_pairplot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
fig = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
execution_timeout: sandbox_timeout
```
- model_plot:

 ![model plot](plots/code_002_pairplot_en_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 1.0
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.444 | expected: scatter_matrix, show | model: scatter_matrix, show | matched: scatter_matrix, show

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
LogisticRegression(max_iter=200)
```
- expected_error:

```python
None
```


**Model output:**
> Here's the generated code:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
```
- model_result:

```python
None
```
- model_error:

```python
None
```


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.linear_model | model: numpy, pandas, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing | matched: sklearn.linear_model
    - calls_score: 0.296 | expected: LogisticRegression, drop, fit | model: LabelEncoder, LogisticRegression, StandardScaler, accuracy_score, fit, fit_transform, predict, train_test_split, transform | matched: LogisticRegression, fit

### code_004_show_all_plots_en (code_generation, language: en, difficulty: hard)

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
execution_timeout: sandbox_timeout
```
- expected_plot:

 ![expected plot](plots/code_004_show_all_plots_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
fig_hist_sepal_length = px.histogram(df, x='sepal_length', color='species')
fig_hist_sepal_width = px.histogram(df, x='sepal_width', color='species')
fig_hist_petal_length = px.histogram(df, x='petal_length', color='species')
fig_hist_petal_width = px.histogram(df, x='petal_width', color='species')
fig_scatter_matrix = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
fig_box_sepal_length = px.box(df, x='species', y='sepal_length')
fig_box_sepal_width = px.box(df, x='species', y='sepal_width')
fig_box_petal_length = px.box(df, x='species', y='petal_length')
fig_box_petal_width = px.box(df, x='species', y='petal_width')
fig_scatter_sepal = px.scatter(df, x='sepal_length', y='sepal_width', color='species')
fig_scatter_petal = px.scatter(df, x='petal_length', y='petal_width', color='species')
fig_violin_sepal_length = px.violin(df, x='species', y='sepal_length', box=True, points='all')
fig_violin_petal_length = px.violin(df, x='species', y='petal_length', box=True, points='all')
fig_parallel_coords = px.parallel_coordinates(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
fig_pairwise = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
fig_hist_sepal_length
fig_hist_sepal_width
fig_hist_petal_length
fig_hist_petal_width
fig_scatter_matrix
fig_box_sepal_length
fig_box_sepal_width
fig_box_petal_length
fig_box_petal_width
fig_scatter_sepal
fig_scatter_petal
fig_violin_sepal_length
fig_violin_petal_length
fig_parallel_coords
fig_pairwise
```
- model_result:

```python
None
```
- model_error:

```python
execution_error:
    Invalid element(s) received for the 'color' property of parcoords.line
        Invalid elements include: ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']

    The 'color' property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color: see https://plotly.com/python/css-colors/ for a list
      - A number that will be interpreted as a color
        according to parcoords.line.colorscale
      - A list or array of any of the above
```
- model_plot_error:
    Invalid element(s) received for the 'color' property of parcoords.line
        Invalid elements include: ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']

    The 'color' property is a color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color: see https://plotly.com/python/css-colors/ for a list
      - A number that will be interpreted as a color
        according to parcoords.line.colorscale
      - A list or array of any of the above


**Метрики:**
- decision_score: True
  - heuristic_score: 0.646
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.111 | expected: plotly.express, plotly.graph_objects | model: plotly.express | matched: plotly.express
    - calls_score: 0.202 | expected: ValueError, box, corr, histogram, imshow, next, scatter, scatter_matrix, select_dtypes, show, tolist | model: box, histogram, parallel_coordinates, scatter, scatter_matrix, show, violin | matched: box, histogram, scatter, scatter_matrix, show

### code_005_histogram_en (code_generation, language: en, difficulty: hard)

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
execution_timeout: sandbox_timeout
```
- expected_plot:

 ![expected plot](plots/code_005_histogram_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
fig = px.histogram(df, x='sepal_length')
fig.show()
```
- model_result:

```python
None
```
- model_error:

```python
execution_timeout: sandbox_timeout
```
- model_plot:

 ![model plot](plots/code_005_histogram_en_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.733
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.178 | expected: ValueError, histogram, select_dtypes, show, tolist | model: histogram, show | matched: histogram, show

### chat_001_what_is_iris_en (chat_response, language: en, difficulty: easy)

**User input:** Explain what this dataset represents.


**Ground truth:**
- expected_facts: dataset contains measurements of petals and sepals; three iris species


**Model output:**
> The data set contains measurements of **iris flowers**.    - **Sepal length** – the long part of the flower’s outer leaf, measured in centimeters.   - **Sepal width** – the width of that outer leaf, also in centimeters.   - **Petal length** – the length of the inner leaf (the petal), in centimeters.   - **Petal width** – the width of the petal, in centimeters.   - **Species** – the type of iris plant, given as a categorical label (for example, *setosa*, *versicolor*, or *virginica*).  In short, each row records the size of a flower’s parts and tells which species it belongs to. This data is often used to practice classification and visualisation techniques.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.663
  - expected_coverage: 0.5 | forbidden_penalty: 0.0
  - покрытые факты: dataset contains measurements of petals and sepals (score 0.656)

### chat_002_how_many_classes_en (chat_response, language: en, difficulty: easy)

**User input:** How many classes are in this dataset?


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> **Answer**  The dataset contains **three classes** in the *species* column.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.27
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features_en (chat_response, language: en, difficulty: medium)

**User input:** Which features best separate the iris species?


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> Here's the generated code:
```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
le = LabelEncoder()
target = le.fit_transform(df['species'])
model = RandomForestClassifier(random_state=0)
model.fit(features, target)
importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
importances
```
- model_result:

```python
None
```
- model_error:

```python
None
```


**Метрики:**
- decision_score: False
- semantic_similarity: 0.063
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class_en (chat_response, language: en, difficulty: medium)

**User input:** Which iris species is the easiest to separate from the others?


**Ground truth:**
- expected_facts: setosa; shorter petals; wider petals


**Model output:**
> Here's the generated code:
```python
import pandas as pd
import numpy as np

centroids = df.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].mean()
dist_matrix = pd.DataFrame(
    np.linalg.norm(centroids.values[:, None, :] - centroids.values[None, :, :], axis=2),
    index=centroids.index,
    columns=centroids.index,
)
min_distances = dist_matrix.replace(0, np.nan).min()
most_separated_species = min_distances.idxmax()
most_separated_species
```
- model_result:

```python
None
```
- model_error:

```python
None
```


**Метрики:**
- decision_score: False
- semantic_similarity: 0.034
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting_en (chat_response, language: en, difficulty: easy)

**User input:** What is overfitting in machine learning?


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> **Overfitting** happens when a machine‑learning model learns the training data **too well**. It captures noise and random quirks instead of the true underlying patterns. As a result, the model performs great on the data it was trained on but **fails on new, unseen data**.  - The model’s error is low on the training set but high on validation or test sets.   - It may look overly complex, with many parameters or deep trees.   - It “remembers” the training examples rather than “understanding” the general rule.  **How to avoid overfitting**  - **Simplify the model** – use fewer features or a less complex algorithm.   - **Use more data** – larger training sets make random noise less influential.   - **Regularization** – techniques like L1/L2 penalties that shrink model weights.   - **Cross‑validation** – test the model on unseen folds while training.   - **Early stopping** – halt training when performance on a validation set stops improving.    In short, overfitting is when a model is *too* fitted to its training data, and it loses the ability to generalize to new situations.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.209
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split_en (chat_response, language: en, difficulty: easy)

**User input:** Why is a train-test split used?


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> A **train‑test split** is a simple way to check how well a model will work on new, unseen data.  - **Training set** – the part of the data the model learns from. It adjusts its parameters to capture patterns. - **Test set** – a separate portion that the model never sees during training. After training, we evaluate the model on this set.  Why we do it:  - **Detect overfitting** – if a model performs well on the training data but poorly on the test data, it has memorized the training examples instead of learning general rules. - **Estimate real‑world performance** – the test set mimics future data, so the accuracy we see there gives a realistic expectation for future predictions. - **Guide model selection** – by comparing results on the same test set, we can choose the best algorithm or hyper‑parameters.  In short, the split creates a fair, unbiased check on how a model will behave once it’s deployed.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.167
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization_en (chat_response, language: en, difficulty: medium)

**User input:** What is regularization and why is it used?


**Ground truth:**
- expected_facts: adds penalty to model complexity; reduces overfitting
- forbidden_facts: regularization increases model variance


**Model output:**
> **Regularization** is a technique that adds a penalty to a model’s complexity during training.    - It discourages the model from fitting the training data too closely.   - By shrinking the size of the model’s coefficients, it makes the model simpler and smoother.    **Why we use it**  - **Prevent over‑fitting:** The model stays general enough to work well on new, unseen data.   - **Improve stability:** Small changes in the data cause only small changes in the learned parameters.   - **Handle multicollinearity:** When features are highly correlated, regularization reduces their impact and keeps the solution well‑behaved.    In short, regularization helps us build models that predict accurately not just on the data we trained on, but also on future data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: regularization increases model variance (score 0.644)

### theory_004_bias_variance_en (chat_response, language: en, difficulty: medium)

**User input:** Explain the bias-variance tradeoff.


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> **Bias–Variance Tradeoff**  When a model learns from data, two kinds of errors can affect its predictions:  - **Bias** – error from wrong assumptions in the model.     High bias means the model is too simple and cannot capture the true patterns. It often **under‑fits** the data.  - **Variance** – error from sensitivity to small fluctuations in the training set.     High variance means the model is too flexible and changes a lot with each new sample. It often **over‑fits** the data.  The tradeoff is that reducing one usually increases the other:  - Making the model more complex (e.g., adding many features or using deep trees) lowers bias but raises variance.   - Simplifying the model (e.g., using a linear regression) lowers variance but raises bias.  **Goal:** Find a balance where both bias and variance are low enough that the overall prediction error on new, unseen data is minimized.  **How to manage the tradeoff**  - Use **cross‑validation** to see how well the model generalizes.   - Apply **regularization** (like Lasso or Ridge) to temper complexity.   - Choose model size or depth that gives the best validation performance.   - Gather more data; larger training sets can reduce variance without increasing bias.  Remember, the sweet spot is where the model captures the underlying patterns **without** being swayed by random noise.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.484
  - expected_coverage: 0.333 | forbidden_penalty: 0.0
  - покрытые факты: tradeoff between bias and variance (score 0.591)

### chat_005_what_is_in_data_en (chat_response, language: en, difficulty: easy)

**User input:** What is in the data?


**Ground truth:**
- expected_facts: The data represents the Iris dataset; The dataset contains numeric features describing sepal and petal measurements; There is a categorical feature indicating the iris species; Each row corresponds to a single flower
- forbidden_facts: Mentioning specific statistics or numeric values; Claims about model performance or training results; Assumptions about downstream data processing


**Model output:**
> The dataset contains five columns.    - **sepal_length** – a numeric value (float) that measures the length of the sepal.   - **sepal_width** – a numeric value (float) that measures the width of the sepal.   - **petal_length** – a numeric value (float) that measures the length of the petal.   - **petal_width** – a numeric value (float) that measures the width of the petal.   - **species** – a categorical label that indicates the species of the flower.    So the data records measurements of iris flower parts and the corresponding flower species.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.676
  - expected_coverage: 0.5 | forbidden_penalty: 0.0
  - покрытые факты: The data represents the Iris dataset (score 0.609); The dataset contains numeric features describing sepal and petal measurements (score 0.614)
