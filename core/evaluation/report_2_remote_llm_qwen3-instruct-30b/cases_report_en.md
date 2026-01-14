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
df.describe(include='all')
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
px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
```
- model_result:

```python
Figure({
    'data': [{'dimensions': [{'axis': {'matches': True},
                              'label': 'sepal_length',
                              'values': {'bdata': ('ZmZmZmZmFECamZmZmZkTQM3MzMzMzB' ... 'ZmZhJAMzMzMzMzFUAAAAAAAAAUQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'sepal_width',
                              'values': {'bdata': ('AAAAAAAADEAAAAAAAAAIQJqZmZmZmQ' ... 'mZmQlAmpmZmZmZDUBmZmZmZmYKQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'petal_length',
                              'values': {'bdata': ('ZmZmZmZm9j9mZmZmZmb2P83MzMzMzP' ... 'ZmZvY/AAAAAAAA+D9mZmZmZmb2Pw=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'petal_width',
                              'values': {'bdata': ('mpmZmZmZyT+amZmZmZnJP5qZmZmZmc' ... 'mZmck/mpmZmZmZyT+amZmZmZnJPw=='),
                                         'dtype': 'f8'}}],
              'hovertemplate': ('species=setosa<br>%{xaxis.titl' ... 'itle.text}=%{y}<extra></extra>'),
              'legendgroup': 'setosa',
              'marker': {'color': '#636efa', 'symbol': 'circle'},
              'name': 'setosa',
              'showlegend': True,
              'type': 'splom'},
             {'dimensions': [{'axis': {'matches': True},
                              'label': 'sepal_length',
                              'values': {'bdata': ('AAAAAAAAHECamZmZmZkZQJqZmZmZmR' ... 'zMzBhAZmZmZmZmFEDNzMzMzMwWQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'sepal_width',
                              'values': {'bdata': ('mpmZmZmZCUCamZmZmZkJQM3MzMzMzA' ... 'MzMwdAAAAAAAAABEBmZmZmZmYGQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'petal_length',
                              'values': {'bdata': ('zczMzMzMEkAAAAAAAAASQJqZmZmZmR' ... 'MzMxFAAAAAAAAACEBmZmZmZmYQQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'petal_width',
                              'values': {'bdata': ('ZmZmZmZm9j8AAAAAAAD4PwAAAAAAAP' ... 'zMzPQ/mpmZmZmZ8T/NzMzMzMz0Pw=='),
                                         'dtype': 'f8'}}],
              'hovertemplate': ('species=versicolor<br>%{xaxis.' ... 'itle.text}=%{y}<extra></extra>'),
              'legendgroup': 'versicolor',
              'marker': {'color': '#EF553B', 'symbol': 'circle'},
              'name': 'versicolor',
              'showlegend': True,
              'type': 'splom'},
             {'dimensions': [{'axis': {'matches': True},
                              'label': 'sepal_length',
                              'values': {'bdata': ('MzMzMzMzGUAzMzMzMzMXQGZmZmZmZh' ... 'AAABpAzczMzMzMGECamZmZmZkXQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'sepal_width',
                              'values': {'bdata': ('ZmZmZmZmCkCamZmZmZkFQAAAAAAAAA' ... 'AAAAhAMzMzMzMzC0AAAAAAAAAIQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'petal_length',
                              'values': {'bdata': ('AAAAAAAAGEBmZmZmZmYUQJqZmZmZmR' ... 'zMzBRAmpmZmZmZFUBmZmZmZmYUQA=='),
                                         'dtype': 'f8'}},
                             {'axis': {'matches': True},
                              'label': 'petal_width',
                              'values': {'bdata': ('AAAAAAAABEBmZmZmZmb+P83MzMzMzA' ... 'AAAABAZmZmZmZmAkDNzMzMzMz8Pw=='),
                                         'dtype': 'f8'}}],
              'hovertemplate': ('species=virginica<br>%{xaxis.t' ... 'itle.text}=%{y}<extra></extra>'),
              'legendgroup': 'virginica',
              'marker': {'color': '#00cc96', 'symbol': 'circle'},
              'name': 'virginica',
              'showlegend': True,
              'type': 'splom'}],
    'layout': {'dragmode': 'select',
               'legend': {'title': {'text': 'species'}, 'tracegroupgap': 0},
               'margin': {'t': 60},
               'template': '...'}
})
```
- model_error:

```python
None
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
1.0
```
- model_error:

```python
None
```


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.linear_model | model: sklearn.linear_model, sklearn.metrics, sklearn.model_selection | matched: sklearn.linear_model
    - calls_score: 0.296 | expected: LogisticRegression, drop, fit | model: LogisticRegression, accuracy_score, fit, predict, train_test_split | matched: LogisticRegression, fit

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
- possible_code_objects: df, pd, np, plt
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
execution_timeout: sandbox_timeout
```
- model_plot:

 ![model plot](plots/code_004_show_all_plots_en_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.566
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.111 | expected: plotly.express, plotly.graph_objects | model: plotly.express | matched: plotly.express
    - calls_score: 0.121 | expected: ValueError, box, corr, histogram, imshow, next, scatter, scatter_matrix, select_dtypes, show, tolist | model: box, scatter, show | matched: box, scatter, show

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
px.histogram(df, x='sepal_length', nbins=20)
```
- model_result:

```python
Figure({
    'data': [{'bingroup': 'x',
              'hovertemplate': 'sepal_length=%{x}<br>count=%{y}<extra></extra>',
              'legendgroup': '',
              'marker': {'color': '#636efa', 'pattern': {'shape': ''}},
              'name': '',
              'nbinsx': 20,
              'orientation': 'v',
              'showlegend': False,
              'type': 'histogram',
              'x': {'bdata': ('ZmZmZmZmFECamZmZmZkTQM3MzMzMzB' ... 'AAAAAAGkDNzMzMzMwYQJqZmZmZmRdA'),
                    'dtype': 'f8'},
              'xaxis': 'x',
              'yaxis': 'y'}],
    'layout': {'barmode': 'relative',
               'legend': {'tracegroupgap': 0},
               'margin': {'t': 60},
               'template': '...',
               'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'sepal_length'}},
               'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'count'}}}
})
```
- model_error:

```python
None
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
> This dataset contains measurements of flower parts, specifically:  - **Sepal length**: The length of the sepal in centimeters. - **Sepal width**: The width of the sepal in centimeters. - **Petal length**: The length of the petal in centimeters. - **Petal width**: The width of the petal in centimeters. - **Species**: The type of flower, such as setosa, versicolor, or virginica.  These measurements are commonly used to classify different species of iris flowers.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.656
  - expected_coverage: 0.5 | forbidden_penalty: 0.0
  - покрытые факты: dataset contains measurements of petals and sepals (score 0.687)

### chat_002_how_many_classes_en (chat_response, language: en, difficulty: easy)

**User input:** How many classes are in this dataset?


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> The dataset has **three classes**.  These are the species of flowers: - **setosa** - **versicolor** - **virginica**


**Метрики:**
- decision_score: True
- semantic_similarity: 0.421
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features_en (chat_response, language: en, difficulty: medium)

**User input:** Which features best separate the iris species?


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> Here's the generated code:
```python
df.groupby('species').agg(['mean', 'std'])
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
> The **setosa** species is the easiest to separate from the others.  It has distinct features: - **Smaller petals** compared to the other species. - **Shorter sepal length and width** on average.  These differences make setosa stand out clearly when looking at the data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.448
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting_en (chat_response, language: en, difficulty: easy)

**User input:** What is overfitting in machine learning?


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> **Overfitting** happens when a machine learning model learns the training data too well — including its noise and outliers — instead of finding the general patterns.  This means the model performs very well on the training data but poorly on new, unseen data.  Think of it like memorizing answers for a test instead of understanding the material. The model "remembers" the training examples but can't apply what it learned to new situations.  **Common signs of overfitting:** - High accuracy on training data - Low accuracy on test or validation data  To prevent overfitting, you can use techniques like: - Simplifying the model - Using more training data - Applying regularization - Using cross-validation  The goal is to build a model that generalizes well to new data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: overfitting improves performance on unseen data (score 0.572)

### theory_002_train_test_split_en (chat_response, language: en, difficulty: easy)

**User input:** Why is a train-test split used?


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> A **train-test split** is used to check how well a model will work on new, unseen data.  Here’s why:  - The model learns from the **training data**. - Then, it is tested on the **test data**, which it hasn’t seen before. - This helps us see if the model is truly learning patterns or just memorizing the training data.  If a model does well on training data but poorly on test data, it might be **overfitting**.  Using a train-test split gives a more honest estimate of the model’s real-world performance.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.126
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization_en (chat_response, language: en, difficulty: medium)

**User input:** What is regularization and why is it used?


**Ground truth:**
- expected_facts: adds penalty to model complexity; reduces overfitting
- forbidden_facts: regularization increases model variance


**Model output:**
> **Regularization** is a technique used in machine learning to prevent a model from becoming too complex.  It works by adding a penalty to the size of the coefficients in the model.  This helps reduce overfitting, which happens when a model learns the training data too well — including its noise and outliers — and performs poorly on new data.  Common types of regularization include: - **L1 regularization** (Lasso): Adds the absolute value of coefficients as a penalty. - **L2 regularization** (Ridge): Adds the squared value of coefficients as a penalty.  Regularization improves a model's ability to generalize to unseen data.  It is especially useful when there are many features or when features are highly correlated.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: regularization increases model variance (score 0.609)

### theory_004_bias_variance_en (chat_response, language: en, difficulty: medium)

**User input:** Explain the bias-variance tradeoff.


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> The **bias-variance tradeoff** is a key concept in machine learning that helps us understand how well a model will perform on new data.  - **Bias** is the error from incorrect assumptions in the model. High bias means the model is too simple and doesn't capture the patterns in the data. This leads to **underfitting**.  - **Variance** is the error from too much sensitivity to small changes in the training data. High variance means the model is too complex and learns noise instead of the true pattern. This leads to **overfitting**.  - A good model balances both:   - Low bias: The model fits the training data well.   - Low variance: The model generalizes well to new data.  Think of it like aiming at a target: - High bias: All shots are far from the center, but close to each other. - High variance: Shots are scattered all over, even if some are close. - Low bias and low variance: Shots are close to the center and close to each other.  The goal is to find the sweet spot where the model is accurate and stable.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.521
  - expected_coverage: 0.333 | forbidden_penalty: 0.0
  - покрытые факты: tradeoff between bias and variance (score 0.619)

### chat_005_what_is_in_data_en (chat_response, language: en, difficulty: easy)

**User input:** What is in the data?


**Ground truth:**
- expected_facts: The data represents the Iris dataset; The dataset contains numeric features describing sepal and petal measurements; There is a categorical feature indicating the iris species; Each row corresponds to a single flower
- forbidden_facts: Mentioning specific statistics or numeric values; Claims about model performance or training results; Assumptions about downstream data processing


**Model output:**
> The data contains measurements of iris flowers.  Each entry has five pieces of information:  - **sepal_length**: The length of the sepal in centimeters. - **sepal_width**: The width of the sepal in centimeters. - **petal_length**: The length of the petal in centimeters. - **petal_width**: The width of the petal in centimeters. - **species**: The type of iris flower. This is a category, like setosa, versicolor, or virginica.  These measurements help identify which species the flower belongs to.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.665
  - expected_coverage: 0.5 | forbidden_penalty: 0.0
  - покрытые факты: The data represents the Iris dataset (score 0.641); There is a categorical feature indicating the iris species (score 0.618)
