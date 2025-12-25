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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['species'])
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
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
    - imports_score: 0.222 | expected: sklearn.linear_model | model: sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing | matched: sklearn.linear_model
    - calls_score: 0.296 | expected: LogisticRegression, drop, fit | model: LabelEncoder, LogisticRegression, accuracy_score, fit, fit_transform, predict, train_test_split | matched: LogisticRegression, fit

### chat_001_what_is_iris (chat_response, language: ru, difficulty: easy)

**User input:** Поясни, что представляет собой датасет.


**Ground truth:**
- expected_facts: датасет содержит измерения лепестков и чашелистиков; три вида ириса
- forbidden_facts: животные


**Model output:**
> Этот датасет содержит измерения цветков ириса.   Он включает следующие столбцы:  - **sepal_length** — длина чашелистика в сантиметрах   - **sepal_width** — ширина чашелистика в сантиметрах   - **petal_length** — длина лепестка в сантиметрах   - **petal_width** — ширина лепестка в сантиметрах   - **species** — вид ириса (категория: например, setosa, versicolor, virginica)  Это классический набор данных, часто используемый для обучения и тестирования алгоритмов машинного обучения, особенно для задач классификации.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.173
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_002_how_many_classes (chat_response, language: ru, difficulty: easy)

**User input:** Сколько классов в этом датасете?


**Ground truth:**
- expected_facts: три класса


**Model output:**
> В этом датасете **три класса**.  Они соответствуют видам цветков в столбце `species`.   Обычно это:   - **setosa**   - **versicolor**   - **virginica**


**Метрики:**
- decision_score: True
- semantic_similarity: 0.284
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features (chat_response, language: ru, difficulty: medium)

**User input:** Какие признаки лучше всего разделяют виды ирисов?


**Ground truth:**
- expected_facts: длина лепестка; ширина лепестка


**Model output:**
> **Пetal length** и **petal width** лучше всего разделяют виды ирисов.  Эти признаки показывают наибольшие различия между видами. Например: - **Setosa** имеет маленькие лепестки. - **Versicolor** средние. - **Virginica** самые большие.  Если смотреть на данные, **длина и ширина лепестков** наиболее сильно отличаются между видами.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.181
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class (chat_response, language: ru, difficulty: medium)

**User input:** Какой вид ириса лучше всего отделяется от остальных?


**Ground truth:**
- expected_facts: setosa; короче лепестки; шире лепестки


**Model output:**
> **Ирис Вирджиния** лучше всего отделяется от остальных видов.  Он имеет более длинные лепестки и наружные чашелистики по сравнению с другими видами. Это делает его легко различимым по внешнему виду.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.603
  - expected_coverage: 0.667 | forbidden_penalty: 0.0
  - покрытые факты: короче лепестки (score 0.568); шире лепестки (score 0.566)

### theory_001_overfitting (chat_response, language: ru, difficulty: easy)

**User input:** Что такое переобучение в машинном обучении?


**Ground truth:**
- expected_facts: модель слишком хорошо подстраивается под обучающую выборку; плохо обобщает на новые данные
- forbidden_facts: переобучение улучшает качество на новых данных


**Model output:**
> **Переобучение** — это когда модель машинного обучения слишком хорошо учится на тренировочных данных.  В результате она запоминает не только общие закономерности, но и шум или случайности в данных.  Когда такая модель сталкивается с новыми, ранее не виденными данными, она начинает плохо работать.  Например, представьте, что вы учитесь на карточках с вопросами. Если вы просто запоминаете ответы, а не понимаете тему, вы не справитесь с новыми вопросами.  В машинном обучении переобучение — это как если бы модель "запомнила" тренировочные примеры, а не научилась обобщать.  Чтобы избежать переобучения, используют методы, такие как: - Разделение данных на тренировочную и проверочную выборки - Упрощение модели - Регуляризация  Цель — чтобы модель хорошо работала как на известных, так и на новых данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.211
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split (chat_response, language: ru, difficulty: easy)

**User input:** Зачем используется разделение на обучающую и тестовую выборки?


**Ground truth:**
- expected_facts: оценка качества модели на невидимых данных; предотвращение утечки данных
- forbidden_facts: для увеличения точности на обучающей выборке


**Model output:**
> Разделение на обучающую и тестовую выборки нужно, чтобы проверить, насколько хорошо модель учится на данных.  - **Обучающая выборка** — это данные, на которых модель учится. - **Тестовая выборка** — это данные, которые модель ещё не видела. Они помогают оценить, насколько хорошо модель работает в реальных условиях.  Если модель будет проверяться только на тех данных, на которых она училась, она может просто запомнить ответы. Но на новых данных она может плохо сработать.  **Главная цель** — убедиться, что модель действительно учится, а не просто запоминает. Это помогает избежать переобучения.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.174
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization (chat_response, language: ru, difficulty: medium)

**User input:** Что такое регуляризация и зачем она нужна?


**Ground truth:**
- expected_facts: добавляет штраф за сложность модели; уменьшает переобучение
- forbidden_facts: регуляризация увеличивает разброс модели


**Model output:**
> **Регуляризация** — это метод, который помогает улучшить работу модели машинного обучения.  Она нужна, чтобы **предотвратить переобучение**. Когда модель слишком хорошо запоминает обучающие данные, она может плохо работать на новых, новых данных.  Регуляризация добавляет небольшое ограничение на размер весов модели. Это заставляет модель быть проще и более обобщённой.  Примеры: - **L1-регуляризация** (Lasso) — может сделать некоторые веса равными нулю. Это помогает отбирать важные признаки. - **L2-регуляризация** (Ridge) — уменьшает размер весов, но не делает их нулевыми.  В итоге, регуляризация помогает модели лучше справляться с новыми данными.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.095
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_004_bias_variance (chat_response, language: ru, difficulty: medium)

**User input:** Объясни компромисс между смещением и разбросом.


**Ground truth:**
- expected_facts: смещение связано с упрощающими предположениями модели; разброс связан с чувствительностью к обучающим данным; существует компромисс между смещением и разбросом
- forbidden_facts: смещение и разброс можно минимизировать независимо


**Model output:**
> Компромисс между **смещением** и **разбросом** — это важная идея в статистике и машинном обучении.  - **Смещение** — это ошибка, которая возникает, когда модель слишком простая. Она не улавливает сложные закономерности в данных.     Например, если данные имеют изгиб, но модель использует прямую линию, она будет сильно ошибаться. Это высокое смещение.  - **Разброс** — это когда модель слишком сложная. Она учится на шуме и случайностях в данных, а не на настоящих закономерностях.     Такая модель хорошо работает на обучающих данных, но плохо — на новых. Это высокий разброс.  Компромисс означает:   - Если вы уменьшаете смещение (делаете модель сложнее), разброс может вырасти.   - Если вы уменьшаете разброс (упрощаете модель), смещение может вырасти.  Цель — найти баланс: модель должна быть достаточно сложной, чтобы улавливать важные паттерны, но не слишком сложной, чтобы не "запоминать" шум.  Пример:   - **Слишком простая модель** — высокое смещение, низкий разброс.   - **Слишком сложная модель** — низкое смещение, высокий разброс.   - **Хороший баланс** — и низкое смещение, и низкий разброс.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.264
  - expected_coverage: 0.0 | forbidden_penalty: 0.0
