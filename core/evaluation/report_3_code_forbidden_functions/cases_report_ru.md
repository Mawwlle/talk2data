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
- expected_error: None


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Пример данных для визуализации
data = {
    'sepal_length': [5.1, 4.9, 4.7, 5.0, 4.8],
    'sepal_width': [3.6, 3.8, 4.2, 3.7, 4.1],
    'petal_length': [1.4, 1.5, 1.6, 1.7, 1.8],
    'petal_width': [0.4, 0.5, 0.6, 0.7, 0.8],
    'species': ['setosa', 'versicolor', 'virginica', 'setosa', 'versicolor']
}

df_data = pd.DataFrame(data)

# Вычисление статистики по датасету
statistics = df_data.describe()

# Вывод статистики
result = f"Статистика по датасету:\n{statistics}"
```
- model_result: None
- model_error:
 
```python
None
```


**Метрики:**
- decision_score: True
- code_score: 0.45
  - heuristic_score: 0.9
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected:  | model: pandas | matched: 
    - calls_score: 0.4 | expected: describe | model: DataFrame, describe | matched: describe
  - result_match: False

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
- expected_error: execution_timeout: sandbox_timeout
- expected_plot:

 ![expected plot](plots/code_002_pairplot_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

# Assuming df is already defined and contains the required columns
pairplot = px.pairplot(df, diag_kind='kde')
pairplot.show()
```
- model_result: None
- model_error:
 
```python
execution_error: module 'plotly.express' has no attribute 'pairplot'
```
- model_plot_error: module 'plotly.express' has no attribute 'pairplot'


**Метрики:**
- decision_score: True
- code_score: 0.35
  - heuristic_score: 0.7
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected: plotly.express | model: plotly.express | matched: plotly.express
    - calls_score: 0.2 | expected: scatter_matrix, show | model: pairplot, show | matched: show
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
- expected_error: None


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Пример данных для обучения классификатора
data = {
    'sepal_length': [5.1, 4.9, 4.7, 5.0, 4.8],
    'sepal_width': [3.6, 3.8, 3.7, 4.2, 4.4],
    'petal_length': [1.4, 1.5, 1.6, 1.7, 1.8],
    'petal_width': [0.4, 0.4, 0.4, 0.4, 0.4],
    'species': ['setosa', 'setosa', 'virginica', 'virginica', 'versicolor']
}

# Создание DataFrame с данными
df_data = pd.DataFrame(data)

# Обучение классификатора
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Разделение данных на тренировочную и тестовую части
X_train, X_test, y_train, y_test = train_test_split(df_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], df_data['species'], test_size=0.2, random_state=42)

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Проверка точности модели
accuracy = model.score(X_test, y_test)
# Вывод результатов
result = f"Accuracy: {accuracy:.2f}"
```
- model_result: None
- model_error:
 
```python
None
```


**Метрики:**
- decision_score: True
- code_score: 0.383
  - heuristic_score: 0.767
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected: sklearn.linear_model | model: pandas, sklearn.linear_model, sklearn.model_selection | matched: sklearn.linear_model
    - calls_score: 0.267 | expected: LogisticRegression, drop, fit | model: DataFrame, LogisticRegression, fit, score, train_test_split | matched: LogisticRegression, fit
  - result_match: False

### chat_001_what_is_iris (chat_response, language: ru, difficulty: easy)

**User input:** Поясни, что представляет собой датасет.


**Ground truth:**
- expected_facts: датасет содержит измерения лепестков и чашелистиков; три вида ириса
- forbidden_facts: животные


**Model output:**
> Датасет представляет собой набор данных, содержащий информацию о различных аспектах животных, такие как их размеры, цвета, и семена.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.094
  - expected_coverage: 0.5 | forbidden_penalty: 1.0
  - покрытые факты: датасет содержит измерения лепестков и чашелистиков (score 0.619)
  - упомянутые запрещённые факты: животные (score 0.616)

### chat_002_how_many_classes (chat_response, language: ru, difficulty: easy)

**User input:** Сколько классов в этом датасете?


**Ground truth:**
- expected_facts: три класса


**Model output:**
> В этом датасете 3 класса.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.812
  - expected_coverage: 1.0 | forbidden_penalty: 0.0
  - покрытые факты: три класса (score 0.688)

### chat_003_best_features (chat_response, language: ru, difficulty: medium)

**User input:** Какие признаки лучше всего разделяют виды ирисов?


**Ground truth:**
- expected_facts: длина лепестка; ширина лепестка


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Пример данных для визуализации
iris_data = {
    'sepal_length': [5.1, 4.9, 4.7, 5.0, 4.8],
    'sepal_width': [3.6, 3.0, 3.2, 3.4, 3.1],
    'petal_length': [1.4, 1.5, 1.6, 1.7, 1.8],
    'petal_width': [0.4, 0.5, 0.6, 0.7, 0.8],
    'species': ['setosa', 'versicolor', 'virginica']
}

# Создание DataFrame
df_iris = pd.DataFrame(iris_data)

# Выбор признаков для разделяения видов ирисов
features_to_split = ['sepal_length', 'sepal_width']

# Разделение данных на признаки и классы
X = df_iris[features_to_split]
y = df_iris['species']

# Определение метода разбиения данных
from sklearn.model_selection import train_test_split

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Вывод результатов
def plot_sepal_length_vs_petal_length():
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train['sepal_length'], X_train['sepal_width'], c=y_train, cmap='viridis')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Sepal Length vs Sepal Width by Species')
    plt.show()

plot_sepal_length_vs_petal_length()
```
- model_result: None
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
> Вид ириса, который лучше всего отделяется от остальных, является квадрат.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.459
  - expected_coverage: 0.333 | forbidden_penalty: 0.0
  - покрытые факты: короче лепестки (score 0.59)

### theory_001_overfitting (chat_response, language: ru, difficulty: easy)

**User input:** Что такое переобучение в машинном обучении?


**Ground truth:**
- expected_facts: модель слишком хорошо подстраивается под обучающую выборку; плохо обобщает на новые данные
- forbidden_facts: переобучение улучшает качество на новых данных


**Model output:**
> Переобучение в машинном обучении - это процесс, когда модель обучается на данных, которые не соответствуют ожиданиям или требованиям, что приводит к переобучению модели. Это может быть связано с различными факторами, таких как переобучение модели на слишком малых данных, переобучение модели на слишком большом наборе данных, переобучение модели на слишком сложных или сложных задачах, переобучение модели на слишком малых или слишком большом количестве признаков.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.206
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split (chat_response, language: ru, difficulty: easy)

**User input:** Зачем используется разделение на обучающую и тестовую выборки?


**Ground truth:**
- expected_facts: оценка качества модели на невидимых данных; предотвращение утечки данных
- forbidden_facts: для увеличения точности на обучающей выборке


**Model output:**
> Зачем используется разделение на обучающую и тестовую выборки? Это делает выборы более эффективным и предсказуемым, особенно при большом количестве данных.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.271
  - expected_coverage: 1.0 | forbidden_penalty: 1.0
  - покрытые факты: оценка качества модели на невидимых данных (score 0.579); предотвращение утечки данных (score 0.662)
  - упомянутые запрещённые факты: для увеличения точности на обучающей выборке (score 0.723)

### theory_003_regularization (chat_response, language: ru, difficulty: medium)

**User input:** Что такое регуляризация и зачем она нужна?


**Ground truth:**
- expected_facts: добавляет штраф за сложность модели; уменьшает переобучение
- forbidden_facts: регуляризация увеличивает разброс модели


**Model output:**
> Регуляризация - это процесс, который устраняет или убирает ошибки в данных, чтобы они были более точными и полезными для анализа. Она помогает улучшить качество данных, устранить неправильные значения или ошибки в данных, которые могут привести к ошибкам при анализе.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: регуляризация увеличивает разброс модели (score 0.557)

### theory_004_bias_variance (chat_response, language: ru, difficulty: medium)

**User input:** Объясни компромисс между смещением и разбросом.


**Ground truth:**
- expected_facts: смещение связано с упрощающими предположениями модели; разброс связан с чувствительностью к обучающим данным; существует компромисс между смещением и разбросом
- forbidden_facts: смещение и разброс можно минимизировать независимо


**Model output:**
> Смещение и разброс - это два различных метода, которые используются для улучшения качества данных. Смещение - это процесс, который позволяет изменить расположение данных в массиве, чтобы улучшить его структуру и качество. Разброс, наоборот, - это процесс, который позволяет изменить распределение данных в массиве, чтобы улучшить его структуру и качество.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.334
  - expected_coverage: 0.0 | forbidden_penalty: 0.0
