# Детальный отчёт по кейсам (en)

## Методика расчёта метрик
### Semantic similarity (только для chat_response)
Формула:
```text
semantic_similarity = 0.6 * similarity + 0.4 * expected_coverage - 0.5 * forbidden_penalty
semantic_similarity = clip(semantic_similarity, 0, 1)
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
Формула:
```text
code_score = 0.5 * heuristic_score + 0.5 * result_match
code_score = clip(code_score, 0, 1)
```
Разложение долей:
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
df.describe()
```
- model_result:
 
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
- model_error:
 
```python
None
```


**Метрики:**
- decision_score: True
- code_score: 1.0
  - heuristic_score: 1.0
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected:  | model:  | matched: 
    - calls_score: 0.444 | expected: describe | model: describe | matched: describe
  - result_match: True

### code_002_pairplot_en (code_generation, language: en, difficulty: medium)

**User input:** Create a pairplot of all features.


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
- code_score: 0.5
  - heuristic_score: 1.0
  - plot_match: 0.0%
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.444 | expected: scatter_matrix, show | model: scatter_matrix, show | matched: scatter_matrix, show
  - result_match: False

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
from sklearn.ensemble import RandomForestClassifier

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
```
- model_result:
 
```python
RandomForestClassifier(random_state=42)
```
- model_error:
 
```python
None
```


**Метрики:**
- decision_score: True
- code_score: 0.241
  - heuristic_score: 0.481
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.0 | expected: sklearn.linear_model | model: sklearn.ensemble, sklearn.model_selection | matched: 
    - calls_score: 0.148 | expected: LogisticRegression, drop, fit | model: RandomForestClassifier, fit, train_test_split | matched: fit
  - result_match: False

### chat_001_what_is_iris_en (chat_response, language: en, difficulty: easy)

**User input:** Explain what this dataset represents.


**Ground truth:**
- expected_facts: dataset contains measurements of petals and sepals; three iris species


**Model output:**
> This dataset represents measurements of iris flowers. It includes four features measured in centimeters: **sepal length**, **sepal width**, **petal length**, and **petal width**. Additionally, it contains a **species** category that identifies the type of iris flower.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.881
  - expected_coverage: 1.0 | forbidden_penalty: 0.0
  - покрытые факты: dataset contains measurements of petals and sepals (score 0.623); three iris species (score 0.564)

### chat_002_how_many_classes_en (chat_response, language: en, difficulty: easy)

**User input:** How many classes are in this dataset?


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> Here's the generated code:
```python
df['species'].nunique()
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
- semantic_similarity: 0.073
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features_en (chat_response, language: en, difficulty: medium)

**User input:** Which features best separate the iris species?


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> To separate the iris species, the features **petal_length** and **petal_width** are generally the most effective. These features help distinguish between different species of iris more clearly than sepal_length and sepal_width.


**Метрики:**
- decision_score: 0
- semantic_similarity: 0.194
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class_en (chat_response, language: en, difficulty: medium)

**User input:** Which iris species is the easiest to separate from the others?


**Ground truth:**
- expected_facts: setosa; shorter petals; wider petals


**Model output:**
> The Iris species that is usually the easiest to separate from the others is **Iris setosa**. This is because it tends to have distinct measurements for sepal length and width compared to the other two species, Iris versicolor and Iris virginica.


**Метрики:**
- decision_score: 0
- semantic_similarity: 0.252
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting_en (chat_response, language: en, difficulty: easy)

**User input:** What is overfitting in machine learning?


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> Overfitting in machine learning occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. Essentially, the model becomes too specialized to the training set and doesn't generalize well to unseen data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: overfitting improves performance on unseen data (score 0.593)

### theory_002_train_test_split_en (chat_response, language: en, difficulty: easy)

**User input:** Why is a train-test split used?


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> A train-test split is used to evaluate the performance of a machine learning model. By dividing the dataset into two parts, one for training and the other for testing, we can ensure that the model learns from the training data and then we can check how well it performs on unseen data in the test set. This helps us understand if the model is generalizing well or just memorizing the training data.


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
> Regularization is a technique used to prevent overfitting in machine learning models. It does this by adding a penalty term to the loss function, which discourages the model from fitting too closely to the training data. This helps the model generalize better to new, unseen data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: regularization increases model variance (score 0.675)

### theory_004_bias_variance_en (chat_response, language: en, difficulty: medium)

**User input:** Explain the bias-variance tradeoff.


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> The bias-variance tradeoff is a fundamental concept in machine learning that describes the balance between two sources of error in predictive models.   - **Bias** refers to the error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss relevant relations between features and target outputs, leading to underfitting. - **Variance** refers to the error due to excessive sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, leading to overfitting.  The goal is to find a model that minimizes both bias and variance to achieve good performance on unseen data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.574
  - expected_coverage: 0.333 | forbidden_penalty: 0.0
  - покрытые факты: tradeoff between bias and variance (score 0.726)
