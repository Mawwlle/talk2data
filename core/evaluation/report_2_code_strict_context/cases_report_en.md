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

**User input:** None


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
- expected_error: None


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Пример данных для визуализации
data = {
    'sepal_length': [5.1, 4.9, 4.7, 5.0, 4.8],
    'sepal_width': [3.6, 3.8, 3.7, 4.2, 4.4],
    'petal_length': [1.4, 1.5, 1.6, 1.7, 1.8],
    'petal_width': [0.4, 0.4, 0.4, 0.4, 0.4],
    'species': ['setosa', 'versicolor', 'virginica', 'setosa', 'versicolor']
}

df = pd.DataFrame(data)

# Вычисление статистики по датасету
statistics = df.describe()

print(statistics)
```
- model_result: None
- model_error:
 
```python
execution_error: name 'print' is not defined
```


**Метрики:**
- decision_score: True
- code_score: 0.45
  - heuristic_score: 0.9
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected:  | model: pandas | matched: 
    - calls_score: 0.4 | expected: describe | model: DataFrame, describe, print | matched: describe
  - result_match: False

### code_002_pairplot_en (code_generation, language: en, difficulty: medium)

**User input:** None


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
- expected_error: execution_timeout: sandbox_timeout
- expected_plot:

 ![expected plot](plots/code_002_pairplot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

# Assuming df is your DataFrame with the specified columns
pairplot = px.pairplot(df, diag_kind='kde')

# Display the pairplot
print(pairplot)
```
- model_result: None
- model_error:
 
```python
execution_error: module 'plotly.express' has no attribute 'pairplot'
```
- model_plot_error: module 'plotly.express' has no attribute 'pairplot'


**Метрики:**
- decision_score: True
- code_score: 0.25
  - heuristic_score: 0.5
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.0 | expected: scatter_matrix, show | model: pairplot, print | matched: 
  - result_match: False

### code_003_train_model_en (code_generation, language: en, difficulty: medium)

**User input:** None


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
- expected_error: None


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Пример данных для обучения
data = {
    'sepal_length': [5.1, 4.9, 4.7, 5.0, 4.8],
    'sepal_width': [3.6, 3.8, 3.7, 4.2, 4.4],
    'petal_length': [1.4, 1.5, 1.6, 1.7, 1.8],
    'petal_width': [0.4, 0.4, 0.4, 0.4, 0.4],
    'species': ['setosa', 'setosa', 'virginica', 'virginica', 'versicolor']
}

# Создание DataFrame
df = pd.DataFrame(data)

# Обучение классификатора
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Разделение данных на тренировочную и тестовую части
X_train, X_test, y_train, y_test = train_test_split(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], df['species'], test_size=0.2, random_state=42)

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Проверка точности модели
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```
- model_result: None
- model_error:
 
```python
execution_error: name 'print' is not defined
```


**Метрики:**
- decision_score: True
- code_score: 0.383
  - heuristic_score: 0.767
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected: sklearn.linear_model | model: pandas, sklearn.linear_model, sklearn.model_selection | matched: sklearn.linear_model
    - calls_score: 0.267 | expected: LogisticRegression, drop, fit | model: DataFrame, LogisticRegression, fit, print, score, train_test_split | matched: LogisticRegression, fit
  - result_match: False

### chat_001_what_is_iris (chat_response, language: en, difficulty: easy)

**User input:** None


**Ground truth:**
- expected_facts: dataset contains measurements of petals and sepals; three iris species


**Model output:**
> Датасет представляет собой набор данных, содержащий информацию о различных аспектах животных, такие как их размеры, цвета, и семена.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.079
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_002_how_many_classes (chat_response, language: en, difficulty: easy)

**User input:** None


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> В этом датасете 3 класса.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.067
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_003_best_features (chat_response, language: en, difficulty: medium)

**User input:** None


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Пример данных для визуализации
iris_data = {
    'sepal_length': [5.1, 4.9, 4.7, 5.0, 4.8],
    'sepal_width': [3.6, 3.8, 4.2, 4.4, 4.6],
    'petal_length': [1.4, 1.5, 1.7, 1.8, 2.0],
    'petal_width': [0.4, 0.5, 0.6, 0.7, 1.0],
    'species': ['setosa', 'versicolor', 'virginica']
}

# Создание DataFrame
df = pd.DataFrame(iris_data)

# Выбор признаков для разделяения видов ирисов
feature_to_split = ['sepal_length', 'sepal_width']

# Разделение данных на признаки и классы
X = df[feature_to_split]
y = df['species']

# Определение метода разбиения данных
from sklearn.model_selection import train_test_split

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Вывод результатов
print("Тренировочные данные:")
print(X_train.head())
print("\nТестовые данные:")
print(X_test.head())
print("\nКлассы:")
print(y_train)
print("\nКлассы тестовых данных:")
print(y_test)
```
- model_result: None
- model_error:
 
```python
None
```


**Метрики:**
- decision_score: False
- semantic_similarity: 0.063
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class (chat_response, language: en, difficulty: medium)

**User input:** None


**Ground truth:**
- expected_facts: setosa; shorter petals; wider petals


**Model output:**
> Вид ириса, который лучше всего отделяется от остальных, является квадрат.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.016
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting (chat_response, language: en, difficulty: easy)

**User input:** None


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> Переобучение в машинном обучении - это процесс, когда модель обучается на данных, которые не соответствуют ожиданиям или требованиям, что приводит к переобучению модели. Это может быть связано с различными факторами, таких как переобучение модели на слишком малых данных, переобучение модели на слишком большом наборе данных, переобучение модели на слишком сложных или сложных задачах, переобучение модели на слишком малых или слишком большом количестве признаков.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split (chat_response, language: en, difficulty: easy)

**User input:** None


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> Зачем используется разделение на обучающую и тестовую выборки? Это делает выборы более эффективным и предсказуемым, особенно при большом количестве данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization (chat_response, language: en, difficulty: medium)

**User input:** None


**Ground truth:**
- expected_facts: adds penalty to model complexity; reduces overfitting
- forbidden_facts: regularization increases model variance


**Model output:**
> Регуляризация - это процесс, который устраняет или убирает ошибки в данных, чтобы они были более точными и полезными для анализа. Она помогает улучшить качество данных, устранить неправильные значения или ошибки в данных, которые могут привести к ошибкам при анализе.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_004_bias_variance (chat_response, language: en, difficulty: medium)

**User input:** None


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> Смещение и разброс - это два различных метода, которые используются для улучшения качества данных. Смещение - это процесс, который позволяет изменить расположение данных в массиве, чтобы улучшить его структуру и качество. Разброс, наоборот, - это процесс, который позволяет изменить распределение данных в массиве, чтобы улучшить его структуру и качество.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.034
  - expected_coverage: 0.0 | forbidden_penalty: 0.0
