# Детальный отчёт по кейсам (ru)

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
None
```
- expected_plot:

 ![expected plot](plots/code_002_pairplot_expected.png)


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

 ![model plot](plots/code_002_pairplot_model.png)


**Метрики:**
- decision_score: True
- code_score: 0.5
  - heuristic_score: 1.0
  - plot_match: 0.0%
    - syntax_check: 0.333 (ok=True)
    - imports_score: 0.222 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.444 | expected: scatter_matrix, show | model: scatter_matrix, show | matched: scatter_matrix, show
  - result_match: False

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
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
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

### chat_001_what_is_iris (chat_response, language: ru, difficulty: easy)

**User input:** Поясни, что представляет собой датасет.


**Ground truth:**
- expected_facts: датасет содержит измерения лепестков и чашелистиков; три вида ириса
- forbidden_facts: животные


**Model output:**
> Датасет представляет собой набор данных, который включает в себя несколько характеристик. В данном случае у нас есть данные о цветках ириса, и они включают следующие параметры:  - **sepal_length** — длина чашелистика, измеряется как число с плавающей запятой. - **sepal_width** — ширина чашелистика, также измеряется как число с плавающей запятой. - **petal_length** — длина лепестка, измеряется как число с плавающей запятой. - **petal_width** — ширина лепестка, измеряется как число с плавающей запятой. - **species** — вид ириса, представленный как категория.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.189
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_002_how_many_classes (chat_response, language: ru, difficulty: easy)

**User input:** Сколько классов в этом датасете?


**Ground truth:**
- expected_facts: три класса


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
- semantic_similarity: 0.05
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features (chat_response, language: ru, difficulty: medium)

**User input:** Какие признаки лучше всего разделяют виды ирисов?


**Ground truth:**
- expected_facts: длина лепестка; ширина лепестка


**Model output:**
> В текущем наборе данных лучшими признаками для разделения видов ирисов являются **длина лепестка** и **ширина лепестка**. Эти параметры часто показывают наиболее заметные различия между разными видами ирисов.


**Метрики:**
- decision_score: 0
- semantic_similarity: 0.745
  - expected_coverage: 1.0 | forbidden_penalty: 0.0
  - покрытые факты: длина лепестка (score 0.568); ширина лепестка (score 0.553)

### chat_004_easiest_class (chat_response, language: ru, difficulty: medium)

**User input:** Какой вид ириса лучше всего отделяется от остальных?


**Ground truth:**
- expected_facts: setosa; короче лепестки; шире лепестки


**Model output:**
> В текущем наборе данных о цветках ириса виды разделяются по четырем параметрам: длина и ширина чашелистика, длина и ширина лепестка. Обычно один из видов ириса, **Ирис сетос**, лучше всего отделяется от других благодаря более уникальным значениям этих параметров. Однако для точного ответа на этот вопрос рекомендуется провести анализ данных или использовать методы машинного обучения.


**Метрики:**
- decision_score: 0
- semantic_similarity: 0.177
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting (chat_response, language: ru, difficulty: easy)

**User input:** Что такое переобучение в машинном обучении?


**Ground truth:**
- expected_facts: модель слишком хорошо подстраивается под обучающую выборку; плохо обобщает на новые данные
- forbidden_facts: переобучение улучшает качество на новых данных


**Model output:**
> Переобучение в машинном обучении — это ситуация, когда модель слишком хорошо учится на обучающих данных и начинает плохо предсказывать новые, неизвестные данные. Это происходит, когда модель становится слишком сложной или имеет слишком много параметров по сравнению с количеством данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.355
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split (chat_response, language: ru, difficulty: easy)

**User input:** Зачем используется разделение на обучающую и тестовую выборки?


**Ground truth:**
- expected_facts: оценка качества модели на невидимых данных; предотвращение утечки данных
- forbidden_facts: для увеличения точности на обучающей выборке


**Model output:**
> Разделение данных на обучающую и тестовую выборки позволяет проверить, насколько хорошо модель предсказывает новые данные. Обучающая выборка используется для обучения модели, а тестовая — для оценки её точности на данных, которые она раньше не видела.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.044
  - expected_coverage: 0.5 | forbidden_penalty: 1.0
  - покрытые факты: предотвращение утечки данных (score 0.566)
  - упомянутые запрещённые факты: для увеличения точности на обучающей выборке (score 0.581)

### theory_003_regularization (chat_response, language: ru, difficulty: medium)

**User input:** Что такое регуляризация и зачем она нужна?


**Ground truth:**
- expected_facts: добавляет штраф за сложность модели; уменьшает переобучение
- forbidden_facts: регуляризация увеличивает разброс модели


**Model output:**
> Регуляризация — это метод, используемый в машинном обучении для предотвращения переобучения модели на обучающих данных. Она помогает сделать модель более устойчивой к новым данным, не виденным ранее. Регуляризация достигается путем добавления дополнительного условия или штрафа к функции потерь модели, что ограничивает сложность модели. Это особенно важно, когда у вас есть большое количество параметров в модели или мало обучающих данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.136
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_004_bias_variance (chat_response, language: ru, difficulty: medium)

**User input:** Объясни компромисс между смещением и разбросом.


**Ground truth:**
- expected_facts: смещение связано с упрощающими предположениями модели; разброс связан с чувствительностью к обучающим данным; существует компромисс между смещением и разбросом
- forbidden_facts: смещение и разброс можно минимизировать независимо


**Model output:**
> Компромисс между смещением и разбросом — это ключевой принцип в машинном обучении. **Смещение** происходит, когда модель слишком проста для данных, что приводит к ошибкам обучения. А **разброс** возникает, когда модель слишком сложна и начинает улавливать шум в данных, что приводит к ошибкам на новых данных. Цель состоит в том, чтобы найти баланс, где модель хорошо работает как на тренировочных, так как и на новых данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.317
  - expected_coverage: 0.0 | forbidden_penalty: 0.0
