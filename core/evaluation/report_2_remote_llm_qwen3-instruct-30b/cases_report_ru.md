# Детальный отчёт по кейсам (ru)

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
### code_001_describe (code_generation, language: ru, difficulty: easy)

**User input:** Напиши код, который выводит статистику по датасету.


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

### code_002_pairplot (code_generation, language: ru, difficulty: medium)

**User input:** Построй pairplot всех признаков.


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

 ![expected plot](plots/code_002_pairplot_expected.png)


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

 ![model plot](plots/code_002_pairplot_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 1.0
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.444 | expected: scatter_matrix, show | model: scatter_matrix, show | matched: scatter_matrix, show

### code_003_train_model (code_generation, language: ru, difficulty: medium)

**User input:** Напиши код, который обучает простой классификатор на этих данных.


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

classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
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

### code_004_missing_check (code_generation, language: ru, difficulty: easy)

**User input:** Проверь, есть ли в датасете пропущенные значения.


**Ground truth:**
- expected_code:
```python
df.isnull().sum()
```
- expected_result:

```python
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
dtype: int64
```
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
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
dtype: int64
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
    - calls_score: 0.444 | expected: isnull, sum | model: isnull, sum | matched: isnull, sum

### code_005_fill_missing (code_generation, language: ru, difficulty: easy)

**User input:** Заполни пропущенные значения средним значением по столбцам.


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


**Метрики:**
- decision_score: True
  - heuristic_score: 1.0
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected:  | model:  | matched:
    - calls_score: 0.444 | expected: fillna, mean | model: fillna, mean | matched: fillna, mean

### code_006_boxplot (code_generation, language: ru, difficulty: medium)

**User input:** Построй boxplot для всех числовых признаков.


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
execution_timeout: sandbox_timeout
```
- expected_plot:

 ![expected plot](plots/code_006_boxplot_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
px.box(df, x='species', y='sepal_length', color='species')
px.box(df, x='species', y='sepal_width', color='species')
px.box(df, x='species', y='petal_length', color='species')
px.box(df, x='species', y='petal_width', color='species')
```
- model_result:

```python
Figure({
    'data': [{'alignmentgroup': 'True',
              'hovertemplate': 'species=%{x}<br>petal_width=%{y}<extra></extra>',
              'legendgroup': 'setosa',
              'marker': {'color': '#636efa'},
              'name': 'setosa',
              'notched': False,
              'offsetgroup': 'setosa',
              'orientation': 'v',
              'showlegend': True,
              'type': 'box',
              'x': array(['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                          'setosa'], dtype=object),
              'x0': ' ',
              'xaxis': 'x',
              'y': {'bdata': ('mpmZmZmZyT+amZmZmZnJP5qZmZmZmc' ... 'mZmck/mpmZmZmZyT+amZmZmZnJPw=='),
                    'dtype': 'f8'},
              'y0': ' ',
              'yaxis': 'y'},
             {'alignmentgroup': 'True',
              'hovertemplate': 'species=%{x}<br>petal_width=%{y}<extra></extra>',
              'legendgroup': 'versicolor',
              'marker': {'color': '#EF553B'},
              'name': 'versicolor',
              'notched': False,
              'offsetgroup': 'versicolor',
              'orientation': 'v',
              'showlegend': True,
              'type': 'box',
              'x': array(['versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                          'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor'],
                         dtype=object),
              'x0': ' ',
              'xaxis': 'x',
              'y': {'bdata': ('ZmZmZmZm9j8AAAAAAAD4PwAAAAAAAP' ... 'zMzPQ/mpmZmZmZ8T/NzMzMzMz0Pw=='),
                    'dtype': 'f8'},
              'y0': ' ',
              'yaxis': 'y'},
             {'alignmentgroup': 'True',
              'hovertemplate': 'species=%{x}<br>petal_width=%{y}<extra></extra>',
              'legendgroup': 'virginica',
              'marker': {'color': '#00cc96'},
              'name': 'virginica',
              'notched': False,
              'offsetgroup': 'virginica',
              'orientation': 'v',
              'showlegend': True,
              'type': 'box',
              'x': array(['virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                          'virginica', 'virginica', 'virginica', 'virginica', 'virginica'],
                         dtype=object),
              'x0': ' ',
              'xaxis': 'x',
              'y': {'bdata': ('AAAAAAAABEBmZmZmZmb+P83MzMzMzA' ... 'AAAABAZmZmZmZmAkDNzMzMzMz8Pw=='),
                    'dtype': 'f8'},
              'y0': ' ',
              'yaxis': 'y'}],
    'layout': {'boxmode': 'overlay',
               'legend': {'title': {'text': 'species'}, 'tracegroupgap': 0},
               'margin': {'t': 60},
               'template': '...',
               'xaxis': {'anchor': 'y',
                         'categoryarray': [setosa, versicolor, virginica],
                         'categoryorder': 'array',
                         'domain': [0.0, 1.0],
                         'title': {'text': 'species'}},
               'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'petal_width'}}}
})
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_006_boxplot_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.296 | expected: box, select_dtypes, show | model: box, show | matched: box, show

### code_007_standardize (code_generation, language: ru, difficulty: medium)

**User input:** Стандартизируй все признаки датасета.


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


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.preprocessing | model: sklearn.preprocessing | matched: sklearn.preprocessing
    - calls_score: 0.296 | expected: StandardScaler, drop, fit_transform | model: StandardScaler, fit_transform | matched: StandardScaler, fit_transform

### code_008_normalize (code_generation, language: ru, difficulty: medium)

**User input:** Нормализуй признаки в диапазон от 0 до 1.


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


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.preprocessing | model: sklearn.preprocessing | matched: sklearn.preprocessing
    - calls_score: 0.296 | expected: MinMaxScaler, drop, fit_transform | model: MinMaxScaler, fit_transform | matched: MinMaxScaler, fit_transform

### code_009_pca (code_generation, language: ru, difficulty: medium)

**User input:** Снизь размерность датасета до двух компонент с помощью PCA.


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
df_pca = pd.DataFrame(pca.fit_transform(df.select_dtypes(include=[float])), columns=['PC1', 'PC2'])
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


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.decomposition | model: pandas, sklearn.decomposition | matched: sklearn.decomposition
    - calls_score: 0.296 | expected: PCA, drop, fit_transform | model: DataFrame, PCA, fit_transform, select_dtypes | matched: PCA, fit_transform

### code_010_pca_plot (code_generation, language: ru, difficulty: medium)

**User input:** Построй scatter plot первых двух PCA-компонент с раскраской по видам.


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
execution_timeout: sandbox_timeout
```
- expected_plot:

 ![expected plot](plots/code_010_pca_plot_expected.png)


**Model output:**
> Here's the generated code:
```python
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['species'] = df['species']

px.scatter(df_pca, x='PC1', y='PC2', color='species', title='Scatter plot of first two PCA components')
```
- model_result:

```python
Figure({
    'data': [{'hovertemplate': 'species=setosa<br>PC1=%{x}<br>PC2=%{y}<extra></extra>',
              'legendgroup': 'setosa',
              'marker': {'color': '#636efa', 'symbol': 'circle'},
              'mode': 'markers',
              'name': 'setosa',
              'orientation': 'v',
              'showlegend': True,
              'type': 'scatter',
              'x': {'bdata': ('TCgv2xZ5BcBwHb3qj7YFwCt/ZBanHA' ... 'A3twbAs49zVz1YBMACnPMXe6AFwA=='),
                    'dtype': 'f8'},
              'xaxis': 'x',
              'y': {'bdata': ('EP0gJgFx1D9gqoDk+afGvwBETuqzjc' ... 'dSLc2/4OGs4oaK4j+ATOo1oJK7Pw=='),
                    'dtype': 'f8'},
              'yaxis': 'y'},
             {'hovertemplate': 'species=versicolor<br>PC1=%{x}<br>PC2=%{y}<extra></extra>',
              'legendgroup': 'versicolor',
              'marker': {'color': '#EF553B', 'symbol': 'circle'},
              'mode': 'markers',
              'name': 'versicolor',
              'orientation': 'v',
              'showlegend': True,
              'type': 'scatter',
              'x': {'bdata': ('EFGrYaWO9D94pcUw8tbtP/SOhkXIbf' ... '/7j+Q/CHkYF80B7b9w9VVt1CLTPw=='),
                    'dtype': 'f8'},
              'xaxis': 'x',
              'y': {'bdata': ('sGilptXs5T8ggGMNlF/UP9DwisXrIu' ... 'n1KZI/cH1It+ox6L/AaxZ3V1TWvw=='),
                    'dtype': 'f8'},
              'yaxis': 'y'},
             {'hovertemplate': 'species=virginica<br>PC1=%{x}<br>PC2=%{y}<extra></extra>',
              'legendgroup': 'virginica',
              'marker': {'color': '#00cc96', 'symbol': 'circle'},
              'mode': 'markers',
              'name': 'virginica',
              'orientation': 'v',
              'showlegend': True,
              'type': 'scatter',
              'x': {'bdata': ('JFEO+eE/BEBEG+lfzqT2P7hux9Xz7g' ... 'PCOvw//MYKwUFq/j+wRhWtNj72Pw=='),
                    'dtype': 'f8'},
              'xaxis': 'x',
              'y': {'bdata': ('ACyJGcUrhL+Qu9T3tmXivwBDAF2CAt' ... 'kYMLQ/AMWbc1TbvT+wqiHnHRfSvw=='),
                    'dtype': 'f8'},
              'yaxis': 'y'}],
    'layout': {'legend': {'title': {'text': 'species'}, 'tracegroupgap': 0},
               'template': '...',
               'title': {'text': 'Scatter plot of first two PCA components'},
               'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'PC1'}},
               'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'PC2'}}}
})
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_010_pca_plot_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.926
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express, sklearn.decomposition | model: pandas, plotly.express, sklearn.decomposition | matched: plotly.express, sklearn.decomposition
    - calls_score: 0.37 | expected: DataFrame, PCA, drop, fit_transform, scatter, show | model: DataFrame, PCA, fit_transform, scatter, show | matched: DataFrame, PCA, fit_transform, scatter, show

### code_012_kmeans (code_generation, language: ru, difficulty: medium)

**User input:** Выполни кластеризацию KMeans на три кластера.


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


**Метрики:**
- decision_score: True
  - heuristic_score: 0.852
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.cluster | model: pandas, sklearn.cluster | matched: sklearn.cluster
    - calls_score: 0.296 | expected: KMeans, drop, fit_predict | model: KMeans, fit_predict | matched: KMeans, fit_predict

### code_012_cluster_plot (code_generation, language: ru, difficulty: medium)

**User input:** Построй scatter plot кластеров KMeans по первым двум признакам.


**Ground truth:**
- expected_code:
```python
import plotly.express as px

X = df.drop(columns=["species"])

fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color=clusters.astype(str))
fig.show()
```
- expected_result:

```python
None
```
- expected_output: Интерактивный график (Plotly)
- expected_error:

```python
execution_error: name 'clusters' is not defined
```
- expected_plot_error: name 'clusters' is not defined


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['sepal_length', 'sepal_width']])

fig = px.scatter(df, x='sepal_length', y='sepal_width', color='cluster', symbol='species', title='KMeans Clusters on First Two Features')
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

 ![model plot](plots/code_012_cluster_plot_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.778
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: pandas, plotly.express, sklearn.cluster | matched: plotly.express
    - calls_score: 0.222 | expected: astype, drop, scatter, show | model: KMeans, fit_predict, scatter, show | matched: scatter, show

### code_013_train_test_split (code_generation, language: ru, difficulty: easy)

**User input:** Раздели датасет на обучающую и тестовую выборки.


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


**Метрики:**
- decision_score: True
  - heuristic_score: 1.0
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: sklearn.model_selection | model: sklearn.model_selection | matched: sklearn.model_selection
    - calls_score: 0.444 | expected: drop, train_test_split | model: drop, train_test_split | matched: drop, train_test_split

### code_014_show_all_plots (code_generation, language: ru, difficulty: hard)

**User input:** Отобрази все графики.


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
- possible_code_objects: df, px, go
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

 ![expected plot](plots/code_014_show_all_plots_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

fig = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species', title='Scatter Matrix of Iris Dataset')
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

 ![model plot](plots/code_014_show_all_plots_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.525
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.111 | expected: plotly.express, plotly.graph_objects | model: plotly.express | matched: plotly.express
    - calls_score: 0.081 | expected: ValueError, box, corr, histogram, imshow, next, scatter, scatter_matrix, select_dtypes, show, tolist | model: scatter_matrix, show | matched: scatter_matrix, show

### code_015_histogram (code_generation, language: ru, difficulty: hard)

**User input:** Отобрази график Histogram


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

 ![expected plot](plots/code_015_histogram_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px
px.histogram(df, x='sepal_length', nbins=20, title='Histogram of Sepal Length')
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
               'template': '...',
               'title': {'text': 'Histogram of Sepal Length'},
               'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'sepal_length'}},
               'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'count'}}}
})
```
- model_error:

```python
None
```
- model_plot:

 ![model plot](plots/code_015_histogram_model.png)


**Метрики:**
- decision_score: True
  - heuristic_score: 0.733
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.178 | expected: ValueError, histogram, select_dtypes, show, tolist | model: histogram, show | matched: histogram, show

### chat_001_what_is_iris (chat_response, language: ru, difficulty: easy)

**User input:** Поясни, что представляет собой датасет.


**Ground truth:**
- expected_facts: датасет содержит измерения лепестков и чашелистиков; три вида ириса
- forbidden_facts: животные


**Model output:**
> Этот датасет содержит измерения цветков ириса.   Он состоит из пяти столбцов:  - **sepal_length** — длина чашелистика в сантиметрах   - **sepal_width** — ширина чашелистика в сантиметрах   - **petal_length** — длина лепестка в сантиметрах   - **petal_width** — ширина лепестка в сантиметрах   - **species** — вид ириса, категориальная переменная (например, setosa, versicolor, virginica)  Данные используются для задач классификации: по размерам лепестков и чашелистиков определить, к какому виду принадлежит цветок.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.185
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_002_how_many_classes (chat_response, language: ru, difficulty: easy)

**User input:** Сколько классов в этом датасете?


**Ground truth:**
- expected_facts: три класса


**Model output:**
> В этом датасете **три класса**.  Они соответствуют видам цветков в столбце `species`.   Это:   - **setosa**   - **versicolor**   - **virginica**


**Метрики:**
- decision_score: True
- semantic_similarity: 0.288
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features (chat_response, language: ru, difficulty: medium)

**User input:** Какие признаки лучше всего разделяют виды ирисов?


**Ground truth:**
- expected_facts: длина лепестка; ширина лепестка


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
- semantic_similarity: 0.052
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class (chat_response, language: ru, difficulty: medium)

**User input:** Какой вид ириса лучше всего отделяется от остальных?


**Ground truth:**
- expected_facts: setosa; короче лепестки; шире лепестки


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

fig = px.scatter(df, x='petal_length', y='petal_width', color='species', title='Разделение видов ирисов по длине и ширине лепестка')
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

 ![model plot](plots/chat_004_easiest_class_model.png)


**Метрики:**
- decision_score: False
- semantic_similarity: 0.051
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting (chat_response, language: ru, difficulty: easy)

**User input:** Что такое переобучение в машинном обучении?


**Ground truth:**
- expected_facts: модель слишком хорошо подстраивается под обучающую выборку; плохо обобщает на новые данные
- forbidden_facts: переобучение улучшает качество на новых данных


**Model output:**
> **Переобучение** — это когда модель машинного обучения слишком хорошо учится на тренировочных данных.    Вот что это означает:  - Модель запоминает конкретные примеры из данных, а не учится находить общие паттерны. - Она работает отлично на тренировочных данных, но плохо на новых, не виденных данных. - Это как если бы студент выучил ответы на тест, но не понял тему — он не справится с похожими вопросами.  Пример: ``` Тренировочные данные: цветок с длиной лепестка 5 см — это ирис. Модель запоминает: "всегда классифицируй как ирис, если длина лепестка 5 см". Но на новых данных: лепесток 5.1 см — модель может ошибиться. ```  Чтобы избежать переобучения, используют методы, такие как: - Разделение данных на тренировочную и тестовую выборки. - Упрощение модели. - Использование регуляризации.  Коротко: **переобучение — это когда модель слишком точно "запоминает" данные, а не учится обобщать**.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.171
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split (chat_response, language: ru, difficulty: easy)

**User input:** Зачем используется разделение на обучающую и тестовую выборки?


**Ground truth:**
- expected_facts: оценка качества модели на невидимых данных; предотвращение утечки данных
- forbidden_facts: для увеличения точности на обучающей выборке


**Model output:**
> Разделение на обучающую и тестовую выборки помогает оценить, насколько хорошо модель работает на новых, не виденных данных.  - **Обучающая выборка** — данные, на которых модель учится делать предсказания. - **Тестовая выборка** — данные, которые модель никогда не видела. Они показывают, как модель справится с реальными задачами.  Если модель работает хорошо только на обучающих данных, но плохо на тестовых, значит, она просто запомнила данные, а не научилась обобщать. Это называется **переобучение**.  Такое разделение помогает избежать переобучения и проверить, насколько модель полезна в реальности.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.177
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization (chat_response, language: ru, difficulty: medium)

**User input:** Что такое регуляризация и зачем она нужна?


**Ground truth:**
- expected_facts: добавляет штраф за сложность модели; уменьшает переобучение
- forbidden_facts: регуляризация увеличивает разброс модели


**Model output:**
> **Регуляризация** — это метод, используемый при обучении моделей машинного обучения, чтобы предотвратить **переобучение**.  Когда модель слишком хорошо учится на тренировочных данных, она может плохо работать на новых, не виденных данных. Это называется переобучением.  Регуляризация помогает решить эту проблему, добавляя небольшое ограничение на величину весов модели. Это заставляет модель быть проще и более обобщённой.  Примеры типов регуляризации: - **L1 регуляризация** (Lasso): уменьшает некоторые веса до нуля, помогая в отборе признаков. - **L2 регуляризация** (Ridge): уменьшает веса, но не до нуля, делая их меньше.  Итог: регуляризация делает модель более надёжной и лучше работает на новых данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.091
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_004_bias_variance (chat_response, language: ru, difficulty: medium)

**User input:** Объясни компромисс между смещением и разбросом.


**Ground truth:**
- expected_facts: смещение связано с упрощающими предположениями модели; разброс связан с чувствительностью к обучающим данным; существует компромисс между смещением и разбросом
- forbidden_facts: смещение и разброс можно минимизировать независимо


**Model output:**
> **Смещение** и **разброс** — это два важных понятия при оценке качества модели в машинном обучении.  - **Смещение** — это ошибка, возникающая из-за упрощённой модели.     Если модель слишком простая, она не может захватить сложные закономерности в данных.     Это приводит к **высокому смещению**.  - **Разброс** — это изменчивость модели при обучении на разных наборах данных.     Если модель слишком сложная, она может "переобучаться" на шуме в данных.     Это приводит к **высокому разбросу**.  **Компромисс** означает, что если вы уменьшаете смещение, разброс может увеличиться, и наоборот.   Хорошая модель — это баланс между ними.   Нужно найти такую модель, которая хорошо работает на новых данных, а не только на обучающих.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.283
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_005_feature_count (chat_response, language: ru, difficulty: easy)

**User input:** Сколько признаков содержит этот датасет?


**Ground truth:**
- expected_facts: четыре признака; длина чашелистика; ширина чашелистика; длина лепестка; ширина лепестка
- forbidden_facts: пять признаков; только признаки лепестков


**Model output:**
> Этот датасет содержит **5 признаков**.  - **sepal_length** — длина чашелистика - **sepal_width** — ширина чашелистика - **petal_length** — длина лепестка - **petal_width** — ширина лепестка - **species** — вид цветка  Каждый признак описывает определённую характеристику цветка.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.322
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_006_samples_count (chat_response, language: ru, difficulty: easy)

**User input:** Сколько объектов содержится в этом датасете?


**Ground truth:**
- expected_facts: 150 объектов; 50 объектов на класс
- forbidden_facts: 100 объектов; несбалансированные классы


**Model output:**
> В этом датасете содержится **150 объектов**.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.66
  - expected_coverage: 1.0 | forbidden_penalty: 0.5
  - покрытые факты: 150 объектов (score 0.858); 50 объектов на класс (score 0.82)
  - упомянутые запрещённые факты: 100 объектов (score 0.822)

### chat_007_numeric_only (chat_response, language: ru, difficulty: easy)

**User input:** Все ли признаки в этом датасете являются числовыми?


**Ground truth:**
- expected_facts: все признаки числовые; непрерывные измерения
- forbidden_facts: категориальные признаки; текстовые признаки


**Model output:**
> Нет, не все признаки в этом датасете являются числовыми.  В датасете есть следующие признаки:  - **sepal_length** — числовое значение (float) - **sepal_width** — числовое значение (float) - **petal_length** — числовое значение (float) - **petal_width** — числовое значение (float) - **species** — категориальный признак (category)  Категориальный признак **species** не является числовым. Он описывает вид цветка, например, "setosa", "versicolor", "virginica".  Таким образом, четыре признака — числовые, а один — категориальный.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.217
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_008_need_scaling (chat_response, language: ru, difficulty: medium)

**User input:** Важно ли масштабирование признаков для этого датасета?


**Ground truth:**
- expected_facts: важно для моделей, основанных на расстоянии; помогает алгоритмам вроде kNN или SVM
- forbidden_facts: масштабирование никогда не нужно; масштабирование меняет метки классов


**Model output:**
> **Масштабирование признаков** может быть важно для этого датасета, особенно если вы используете алгоритмы, чувствительные к масштабу данных.  Вот почему:  - Признаки, такие как **sepal_length**, **sepal_width**, **petal_length**, и **petal_width**, измеряются в разных единицах и имеют разные диапазоны значений. - Алгоритмы, такие как **логистическая регрессия**, **метод опорных векторов (SVM)** или **многослойные нейронные сети**, могут работать хуже, если признаки не масштабированы. - **K-ближайших соседей (KNN)** также сильно зависит от масштаба, потому что он использует расстояния между точками.  Однако, если вы используете **деревья решений** или **случайный лес**, масштабирование не обязательно — они не зависят от масштаба признаков.  > **Вывод:** Масштабирование полезно, если вы используете алгоритмы, основанные на расстояниях или градиентных методах.   > Если не уверены — масштабирование безопасно и часто улучшает результаты.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.243
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_009_linear_separable (chat_response, language: ru, difficulty: medium)

**User input:** Является ли этот датасет линейно разделимым?


**Ground truth:**
- expected_facts: класс setosa линейно разделим; остальные классы перекрываются
- forbidden_facts: все классы идеально разделимы; ни один класс не разделим


**Model output:**
> Этот датасет **не является линейно разделимым**.  **Почему?**   Датасет содержит три вида цветков ириса: setosa, versicolor и virginica.   - **Setosa** можно отделить от других линиями.   - Но **versicolor** и **virginica** перекрываются в признаках, как длина и ширина лепестков.    Это означает, что **одна прямая или плоскость не сможет разделить все три класса**.   Нужны более сложные методы, например, деревья решений или нейросети.  Если хочешь, я могу помочь построить визуализацию или выбрать подходящий алгоритм.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.188
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_011_unsupervised_possible (chat_response, language: ru, difficulty: medium)

**User input:** Можно ли использовать этот датасет для обучения без учителя?


**Ground truth:**
- expected_facts: можно использовать для кластеризации; метки классов можно игнорировать
- forbidden_facts: обучение без учителя требует меток


**Model output:**
> **Да**, этот датасет можно использовать для обучения без учителя.  В датасете есть признаки:   - **sepal_length** — длина чашелистика   - **sepal_width** — ширина чашелистика   - **petal_length** — длина лепестка   - **petal_width** — ширина лепестка    Эти признаки — числовые данные, которые подходят для задач кластеризации или снижения размерности, например, с помощью методов, таких как K-средние или t-SNE.  **Обратите внимание**: столбец `species` — это метка класса. Для обучения без учителя его лучше **не использовать**, так как цель — найти скрытые паттерны без знания истинных групп.  Таким образом, вы можете взять только числовые признаки и применить методы без учителя.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.125
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_012_pca_purpose (chat_response, language: ru, difficulty: medium)

**User input:** Применим ли метод PCA к этому датасету?


**Ground truth:**
- expected_facts: снижение размерности; возможность визуализации; сохранение большей части дисперсии
- forbidden_facts: PCA улучшает метки классов


**Model output:**
> **Да**, метод **PCA** (главные компоненты) можно применить к этому датасету.  Вот почему:  - Данные содержат **числовые признаки**: *sepal_length*, *sepal_width*, *petal_length*, *petal_width* — все они числовые и могут быть нормализованы. - Признак *species* — это категориальная переменная, но её **не нужно включать в PCA**, так как PCA работает только с числовыми данными. - PCA полезен для уменьшения размерности и визуализации, особенно если признаков много.  **Важно**: перед применением PCA нужно: - Удалить или не использовать столбец *species*. - Нормализовать данные, чтобы все признаки были в одном масштабе.  > Пример: PCA работает только с числовыми столбцами, поэтому используйте только *sepal_length*, *sepal_width*, *petal_length*, *petal_width*.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.128
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_013_outliers (chat_response, language: ru, difficulty: medium)

**User input:** Являются ли выбросы серьёзной проблемой в этом датасете?


**Ground truth:**
- expected_facts: датасет относительно чистый; мало или нет выраженных выбросов
- forbidden_facts: датасет сильно искажён выбросами


**Model output:**
> **Выбросы** могут быть серьёзной проблемой, если они искажают данные или влияют на анализ.  В этом датасете есть числовые признаки:   - **sepal_length** (длина чашелистика)   - **sepal_width** (ширина чашелистика)   - **petal_length** (длина лепестка)   - **petal_width** (ширина лепестка)  Если какие-то значения сильно отклоняются от остальных, они могут повлиять на результаты моделей или статистики.  Но без анализа данных мы не можем точно сказать, есть ли выбросы.  Чтобы проверить, нужно посмотреть данные — например, с помощью графиков или статистик.  Если хочешь, я помогу сформулировать запрос для генерации кода, чтобы проверить выбросы.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.18
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_014_real_world_limitations (chat_response, language: ru, difficulty: hard)

**User input:** Какие ограничения есть у этого датасета?


**Ground truth:**
- expected_facts: небольшой размер; не отражает сложные реальные данные
- forbidden_facts: подходит для масштабного глубокого обучения


**Model output:**
> Этот датасет имеет несколько ограничений:  - **Только три вида растений**: данные содержат информацию только о трёх видах ирисов — setosa, versicolor и virginica. Это ограничивает обобщение на другие виды растений.  - **Ограниченное количество признаков**: в датасете всего четыре числовых признака — длина и ширина чашелистиков и лепестков. Это не учитывает другие возможные характеристики, например, цвет, текстура или размеры стеблей.  - **Небольшой объём данных**: в датасете всего 150 записей. Это мало для сложных моделей машинного обучения, особенно если нужно обучать на более крупных или разнообразных данных.  - **Данные сбалансированы**: по 50 примеров каждого вида. В реальном мире распределение классов часто несбалансировано, поэтому модель может плохо работать на несбалансированных данных.  - **Данные симулированы или собраны в лаборатории**: вероятно, данные были получены из контролируемых условий. Это может не отражать реальные условия роста растений в природе.  - **Признаки измерены с высокой точностью**: измерения, скорее всего, сделаны с точностью до десятых сантиметра. В реальности измерения могут быть неточными или с разной точностью.  Если вы хотите работать с более сложными или реалистичными данными, стоит рассмотреть другие датасеты.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.167
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_006_classification_vs_regression (chat_response, language: ru, difficulty: easy)

**User input:** В чём разница между классификацией и регрессией?


**Ground truth:**
- expected_facts: классификация предсказывает дискретные классы; регрессия предсказывает непрерывные значения
- forbidden_facts: классификация предсказывает только числа; регрессия предсказывает категории


**Model output:**
> **Классификация** и **регрессия** — это два типа задач в машинном обучении.  - **Классификация** — это когда модель предсказывает **категорию** или **метку**.     Например, определить, какой вид цветка: ирис-сиреневый, ирис-разноцветный или ирис-пурпурный.  - **Регрессия** — это когда модель предсказывает **числовое значение**.     Например, предсказать длину лепестка в сантиметрах.  **Ключевое отличие:**   - Классификация — **категории**.   - Регрессия — **числа**.  Пример из данных:   - Если модель определяет вид ириса — это **классификация**.   - Если модель предсказывает длину лепестка — это **регрессия**.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.29
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_007_cross_validation (chat_response, language: ru, difficulty: medium)

**User input:** Что такое кросс-валидация?


**Ground truth:**
- expected_facts: разбиение данных на несколько фолдов; более надёжная оценка качества модели; уменьшает зависимость от одного train-test разбиения
- forbidden_facts: кросс-валидация гарантирует идеальное качество


**Model output:**
> **Кросс-валидация** — это способ проверки, насколько хорошо модель работает на новых данных.  Вот как это работает:  - Данные делятся на части. - Модель обучается на одной части и проверяется на другой. - Этот процесс повторяется несколько раз, чтобы получить надёжную оценку.  Пример:   Если у нас 1000 примеров, и мы используем **5-кратную кросс-валидацию**, данные делятся на 5 частей.   Модель обучается на 4 частях и проверяется на одной.   Это повторяется 5 раз, каждый раз с другой частью в качестве проверки.  **Цель** — избежать переобучения и получить честную оценку производительности модели.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.225
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_008_feature_engineering (chat_response, language: ru, difficulty: medium)

**User input:** Что такое feature engineering?


**Ground truth:**
- expected_facts: процесс создания или преобразования признаков; может улучшить качество модели; требует знания предметной области
- forbidden_facts: feature engineering — это только очистка данных


**Model output:**
> **Feature engineering** — это процесс создания новых признаков или улучшения существующих из исходных данных, чтобы помочь модели машинного обучения лучше понимать информацию.  Например, в данных о цветах ириса: - У нас есть длина и ширина чашелистиков и лепестков. - Можно создать новый признак, например, **отношение длины лепестка к ширине**. - Это может помочь модели лучше различать виды ирисов.  Цель — сделать данные более полезными для модели.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.236
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_009_data_leakage (chat_response, language: ru, difficulty: medium)

**User input:** Что такое утечка данных?


**Ground truth:**
- expected_facts: информация из тестовой выборки попадает в обучение; приводит к завышенной оценке качества; ухудшает обобщающую способность
- forbidden_facts: утечка данных улучшает качество в реальных условиях


**Model output:**
> **Утечка данных** — это случай, когда конфиденциальная или личная информация попадает к незаконным лицам.  Например: - Кто-то крадёт пароли или данные кредитных карт. - Информация случайно отправляется не тому человеку. - Система взломана, и данные оказались в интернете.  Такие утечки могут повредить репутации компании или нанести вред людям.  Важно защищать данные с помощью паролей, шифрования и безопасных систем.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.248
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_010_precision_recall (chat_response, language: ru, difficulty: medium)

**User input:** Поясни, что такое precision и recall.


**Ground truth:**
- expected_facts: precision показывает точность положительных предсказаний; recall показывает полноту выявления положительных объектов; существует компромисс между precision и recall
- forbidden_facts: precision и recall всегда равны


**Model output:**
> **Precision** и **recall** — это метрики, которые используются для оценки качества модели в задачах классификации.  - **Precision** (точность) показывает, насколько надёжны предсказания модели.     Она отвечает на вопрос: *«Из всех объектов, которые модель отнесла к классу, сколько действительно принадлежат этому классу?»*     Например, если модель говорит, что цветок — это ирис, то precision покажет, насколько часто она права.  - **Recall** (полнота) показывает, насколько хорошо модель находит все объекты нужного класса.     Она отвечает на вопрос: *«Из всех объектов, которые действительно относятся к классу, сколько модель смогла правильно обнаружить?»*     Например, если есть много ирисов, то recall покажет, сколько из них модель смогла распознать.  **Пример:** - Если модель находит 10 ирисов, но только 7 из них — настоящие, то precision = 7 / 10 = 0.7. - Если среди всех ирисов в данных 12, а модель нашла только 7, то recall = 7 / 12 ≈ 0.58.  Иногда нужно выбирать между precision и recall.   Высокий precision — модель не ошибается, но может пропустить много.   Высокий recall — модель находит почти всё, но может ошибаться.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.272
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_011_accuracy_limitations (chat_response, language: ru, difficulty: medium)

**User input:** Является ли accuracy универсальной метрикой?


**Ground truth:**
- expected_facts: проблематична при несбалансированных данных; не отражает качество по отдельным классам
- forbidden_facts: accuracy всегда лучшая метрика


**Model output:**
> **Accuracy** — это метрика, которая показывает, как часто модель даёт правильные предсказания.  Но **accuracy не является универсальной метрикой**.  Вот почему:  - Она может быть вводящей в заблуждение, если классы не сбалансированы. - Например, если 95% данных — один класс, модель может просто предсказывать этот класс и получить accuracy 95%, хотя и не научится различать другие.  **Лучше использовать** accuracy вместе с другими метриками, такими как **precision**, **recall** или **F1-score**, особенно в задачах с несбалансированными данными.  > Пример: В медицинской диагностике важно не только правильно определить болезнь, но и не пропустить её. Там accuracy может быть не самой важной метрикой.  Так что: **accuracy — полезна, но не универсальна**.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.088
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_012_what_is_in_data (chat_response, language: ru, difficulty: easy)

**User input:** Что в данных


**Ground truth:**
- expected_facts: Данные представляют собой датасет Iris; В данных содержатся числовые признаки, описывающие размеры чашелистиков и лепестков; В данных есть категориальный признак с видом ириса; Каждая строка соответствует одному цветку
- forbidden_facts: Упоминание конкретных статистик или числовых значений; Утверждения о качестве модели или результатах обучения; Предположения о последующей обработке данных


**Model output:**
> В данных есть следующие столбцы:  - **sepal_length** — длина чашелистика (в сантиметрах) - **sepal_width** — ширина чашелистика (в сантиметрах) - **petal_length** — длина лепестка (в сантиметрах) - **petal_width** — ширина лепестка (в сантиметрах) - **species** — вид цветка (категория, например, setosa, versicolor, virginica)  Это данные о цветках ириса.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.283
  - expected_coverage: 0.0 | forbidden_penalty: 0.0
