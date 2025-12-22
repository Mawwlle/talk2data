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
- expected_error: None


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Assuming df is already defined in the environment
# Calculate mean, median, standard deviation, and count of each column
mean_sepal_length = df['sepal_length'].mean()
median_sepal_length = df['sepal_length'].median()
std_sepal_length = df['sepal_length'].std()
count_sepal_length = df['sepal_length'].count()

mean_petal_length = df['petal_length'].mean()
median_petal_length = df['petal_length'].median()
std_petal_length = df['petal_length'].std()
count_petal_length = df['petal_length'].count()

# Print the results
result = {
    'Mean Sepal Length': mean_sepal_length,
    'Median Sepal Length': median_sepal_length,
    'Standard Deviation Sepal Length': std_sepal_length,
    'Count Sepal Length': count_sepal_length,

    'Mean Petal Length': mean_petal_length,
    'Median Petal Length': median_petal_length,
    'Standard Deviation Petal Length': std_petal_length,
    'Count Petal Length': count_petal_length
}

# Print the result
for key, value in result.items():
    if isinstance(value, float):
        # Convert to string for clarity
        value_str = str(value)
    else:
        value_str = str(value)
    result[key] = value_str

# Print the final result
for key, value in result.items():
    if isinstance(value, float):
        # Convert to string for clarity
        value_str = str(value)
    else:
        value_str = str(value)
    result[key] = value_str
```
- model_result: None
- model_error:
 
```python
None
```


**Метрики:**
- decision_score: True
- code_score: 0.25
  - heuristic_score: 0.5
    - syntax_check: 0.3 (ok=True)
    - imports_score: 0.2 | expected:  | model: pandas | matched: 
    - calls_score: 0.0 | expected: describe | model: count, isinstance, items, mean, median, std, str | matched: 
  - result_match: False

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
- expected_error: execution_timeout: sandbox_timeout
- expected_plot:

 ![expected plot](plots/code_002_pairplot_en_expected.png)


**Model output:**
> Here's the generated code:
```python
import plotly.express as px

# Assuming df is already defined and contains the required columns
fig = px.pairplot(df, diag_kind='hist')
fig.show()
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
- expected_error: None


**Model output:**
> Here's the generated code:
```python
import pandas as pd

# Assuming df is your DataFrame with the specified columns
# Let's assume you have a column named 'species' which contains the target variable
# For demonstration, let's assume we want to train a simple classifier using a linear model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Print the accuracy of the model
accuracy = model.score(X_test, y_test)
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
    - calls_score: 0.267 | expected: LogisticRegression, drop, fit | model: LogisticRegression, fit, predict, score, train_test_split | matched: LogisticRegression, fit
  - result_match: False

### chat_001_what_is_iris_en (chat_response, language: en, difficulty: easy)

**User input:** Explain what this dataset represents.


**Ground truth:**
- expected_facts: dataset contains measurements of petals and sepals; three iris species


**Model output:**
> This dataset represents a dataset of 150 samples, each containing information about a flower. The dataset includes the following features: - Sepal length - Sepal width - Petal length - Petal width - Species The dataset is used for various purposes, such as: 1. Data analysis: To understand the relationship between different features and the species of flowers. 2. Machine learning: To build models that can predict the species of flowers based on their features. 3. Image classification: To classify images of flowers into different categories. 4. Data visualization: To visualize the dataset and understand its structure. The dataset is available in various formats, including CSV, Excel, and JSON, and can be easily accessed and manipulated using Python libraries such as Pandas.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.562
  - expected_coverage: 0.5 | forbidden_penalty: 0.0
  - покрытые факты: dataset contains measurements of petals and sepals (score 0.659)

### chat_002_how_many_classes_en (chat_response, language: en, difficulty: easy)

**User input:** How many classes are in this dataset?


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> The dataset contains 5 classes.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.352
  - expected_coverage: 0.25 | forbidden_penalty: 0.0
  - покрытые факты: three classes (score 0.572)

### chat_003_best_features_en (chat_response, language: en, difficulty: medium)

**User input:** Which features best separate the iris species?


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> The features that best separate the iris species are sepal length and sepal width.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.18
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class_en (chat_response, language: en, difficulty: medium)

**User input:** Which iris species is the easiest to separate from the others?


**Ground truth:**
- expected_facts: setosa; shorter petals; wider petals


**Model output:**
> The easiest iris species to separate from the others is the Iris versicolor.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.168
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting_en (chat_response, language: en, difficulty: easy)

**User input:** What is overfitting in machine learning?


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> Overfitting is a common problem in machine learning where a model learns the training data too well, making it perform poorly on unseen data. It occurs when a model's performance is significantly improved by training on too many training examples, leading to a poor generalization ability.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: overfitting improves performance on unseen data (score 0.673)

### theory_002_train_test_split_en (chat_response, language: en, difficulty: easy)

**User input:** Why is a train-test split used?


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> A train-test split is used to evaluate the performance of a machine learning model. It is a common technique in machine learning to divide the data into two subsets: one for training the model and another for testing it. This helps in reducing the risk of overfitting by ensuring that the model is not too complex on the training data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.117
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization_en (chat_response, language: en, difficulty: medium)

**User input:** What is regularization and why is it used?


**Ground truth:**
- expected_facts: adds penalty to model complexity; reduces overfitting
- forbidden_facts: regularization increases model variance


**Model output:**
> Regularization is a technique used to prevent overfitting in machine learning models. It involves adding a penalty term to the loss function, which discourages the model from fitting the training data too closely. This penalty term is typically a constant value, often set to a small positive number, such as 1e-5 or 1e-6. Regularization helps prevent the model from becoming too complex and overfitting, which can lead to poor generalization to new data.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.0
  - expected_coverage: 0.0 | forbidden_penalty: 1.0
  - упомянутые запрещённые факты: regularization increases model variance (score 0.657)

### theory_004_bias_variance_en (chat_response, language: en, difficulty: medium)

**User input:** Explain the bias-variance tradeoff.


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> The bias-variance tradeoff is a fundamental concept in statistics that describes how the variance of a dataset can be reduced by adding more data points. It is a measure of the amount of variation in a dataset that can be explained by adding more data points. The tradeoff is often expressed as a trade-off between the amount of variance explained by the additional data points and the amount of variance that can be explained by the original data points.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.075
  - expected_coverage: 0.333 | forbidden_penalty: 1.0
  - покрытые факты: tradeoff between bias and variance (score 0.785)
  - упомянутые запрещённые факты: bias and variance can both be minimized independently (score 0.551)
