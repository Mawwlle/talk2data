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
> I can help you understand how to generate statistics for your dataset using Python. For instance, you could calculate the mean, median, minimum, and maximum values for each feature in your dataset. Here's a simple plan:  1. Import the dataset. 2. Convert categorical columns like `species` to numeric values (if necessary). 3. Calculate the statistics for each numerical feature like `sepal_length` and `petal_length`. 4. Optionally, if you are using Jupyter Notebook, you can display the results in a tabular format.  We can generate Python code for this, but it would require you to provide more context about what exact dataset you're working with.  How would you like to proceed?


**Метрики:**
- decision_score: False
  - heuristic_score: 0.222
    - syntax_check: 0.0 (ok=True)
    - imports_score: 0.222 | expected:  | model:  | matched:
    - calls_score: 0.0 | expected: describe | model:  | matched:

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
> К сожалению, я не могу выполнять запросы, требующие программирования. Но могу помочь генерировать Python-код для построения pairplot всех признаков. Как вы хотите сформулировать вопрос?


**Метрики:**
- decision_score: False
  - heuristic_score: 0.0
    - syntax_check: 0.0 (ok=True)
    - imports_score: 0.0 | expected: plotly.express | model:  | matched:
    - calls_score: 0.0 | expected: scatter_matrix, show | model:  | matched:

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
> It seems like you're asking a technical question, so I'll generate Python code for you instead. Here’s how to build a Simple Neural Network using Python's scikit-learn and TensorFlow libraries for the given dataset:  ```python from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import accuracy_score  import tensorflow as tf  # Import and preprocess data  # Assuming your data is in CSV format:  import pandas as pd  data = pd.read_csv('path_to_your_data.csv') X = data.iloc[:, :-1].values  # feature data except target y = data.iloc[:, -1].values  # target values  # Split dataset into training and test sets X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Standardize the features for better model performance scaler = StandardScaler().fit(X_train) X_train_std = scaler.transform(X_train) X_test_std = scaler.transform(X_test)  # Build the model using Logistic Regression or Random Forest model = LogisticRegression() # model.fit(X_train_std, y_train)  # If using a deep learning model with TensorFlow: model = tf.keras.models.Sequential([     tf.keras.layers.Dense(units=8, activation='relu', input_shape=(X_train_std.shape1],)),     tf.keras.layers.Dropout(0.1),     tf.keras.layers.Dense(units=8, activation='relu'),     tf.keras.layers.Dropout(0.1),     tf.keras.layers.Dense(units=3, activation='softmax')  # adjust to predicted classes  # Train the model using Gradient Descent optimizers model.compile(optimizer='adam',               loss='sparse_categorical_crossentropy',               metrics=['accuracy'])  # Train the model using early stopping to avoid overfitting from tensorflow.keras.callbacks import EarlyStopping early_stopping = EarlyStopping(     monitor='val_loss',     patience=5,     restore_best_weights=True,     verbose=1 )  model.fit(X_train_std,           y_train,           epochs=10,           validation_data=(X_test_std, y_test),           callbacks=[early_stopping])  # Evaluate trained model test_loss, test_acc = model.evaluate(X_test_std, y_test, verbose=1) print(f'Test accuracy: {test_acc*100:.2f}%')  # Make predictions on new data y_pred = model.predict(X_test_std) y_pred = [np.argmax(preds) for preds in y_pred]  # Calculate accuracy using sklearn functions accuracy = accuracy_score(y_test, y_pred) print(f'Accuracy score: {accuracy:.2f}') ```  This code uses a simple neural network or a scikit-learn Logistic Regression model, depending on your choice. It's a common pipeline using a split between train and test datasets and applying StandardScaler for feature scaling. The TensorFlow version can use the Adam optimizer for training.  Which library do you want to use?


**Метрики:**
- decision_score: False
  - heuristic_score: 0.0
    - syntax_check: 0.0 (ok=True)
    - imports_score: 0.0 | expected: sklearn.linear_model | model:  | matched:
    - calls_score: 0.0 | expected: LogisticRegression, drop, fit | model:  | matched:

### chat_001_what_is_iris (chat_response, language: ru, difficulty: easy)

**User input:** Поясни, что представляет собой датасет.


**Ground truth:**
- expected_facts: датасет содержит измерения лепестков и чашелистиков; три вида ириса
- forbidden_facts: животные


**Model output:**
> Датасет содержит информацию о цветах, включая их длину, ширину и другие характеристики. Он помогает анализировать разные виды растений.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.608
  - expected_coverage: 0.5 | forbidden_penalty: 0.0
  - покрытые факты: датасет содержит измерения лепестков и чашелистиков (score 0.637)

### chat_002_how_many_classes (chat_response, language: ru, difficulty: easy)

**User input:** Сколько классов в этом датасете?


**Ground truth:**
- expected_facts: три класса


**Model output:**
> В этом датасете три класса.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.868
  - expected_coverage: 1.0 | forbidden_penalty: 0.0
  - покрытые факты: три класса (score 0.78)

### chat_003_best_features (chat_response, language: ru, difficulty: medium)

**User input:** Какие признаки лучше всего разделяют виды ирисов?


**Ground truth:**
- expected_facts: длина лепестка; ширина лепестка


**Model output:**
> To determine the best признаки (features) that distinguish types of irises, we should analyze the features in the dataset you've provided. Давайте рассмотрим каждую колонку:  - **sepal_length**: Длина лепестков. - **sepal_width**: Ширина лепестков. - **petal_length**: Длина лепестков. - **petal_width**: Ширина лепестков. - **species**: Вид ириса (iris).  Варианты анализа:  1. **sepal_length** и **sepal_width**: Эти параметры обычно не столь полезны, как длина и ширина лепестков. 2. **petal_length** и **petal_width**: Данные показатели могут содержать много информации о форме и окраске цветка, что хорошо подходит для классификации видов ирисов.  Таким образом, учитывая эти соображения, можно сделать вывод, что признаки **petal_length** и **petal_width** лучше всего подойдут для разделения видов ирисов.  Если у вас есть дополнительные вопросы или вы хотели бы использовать эти данные для обучения модели, я могу помочь!


**Метрики:**
- decision_score: True
- semantic_similarity: 0.112
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### chat_004_easiest_class (chat_response, language: ru, difficulty: medium)

**User input:** Какой вид ириса лучше всего отделяется от остальных?


**Ground truth:**
- expected_facts: setosa; короче лепестки; шире лепестки


**Model output:**
> Sorting a dataset can be done in different ways.   1. **One way to sort them**:     - Use the species column, which classifies the type of iris.     - We sort this alphabetically:     ```python     # Sorting the dataset by species alphabetically     sorted(dataset, key=lambda x: x['species'])     ```  2. **Another way**:     - We can sort based on species, then separate based on the dataset.     ```python     # Sorting the dataset by species and separating based on species names     separated_by_species = {}     for index, row in dataset.iterrows():         species_name = row['species']         if species_name not in separated_by_species:             separated_by_speciesspecies_name = []         separated_by_speciesspecies_name].append(row)     ```  Would you like to know how to visualize this data or further filter it based on your request?


**Метрики:**
- decision_score: True
- semantic_similarity: 0.024
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_001_overfitting (chat_response, language: ru, difficulty: easy)

**User input:** Что такое переобучение в машинном обучении?


**Ground truth:**
- expected_facts: модель слишком хорошо подстраивается под обучающую выборку; плохо обобщает на новые данные
- forbidden_facts: переобучение улучшает качество на новых данных


**Model output:**
> Переобучение в машинном обучении происходит, когда модель слишком хорошо "запоминает" обучающие данные, но при этом плохо обобщает новые данные. Это может привести к низкой точности прогнозов на новых данных. Чтобы избежать переобучения, часто используют техники регуляризации, такие как уменьшение размера сети или включение дополнительных контрольных точек в обучение.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.287
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_002_train_test_split (chat_response, language: ru, difficulty: easy)

**User input:** Зачем используется разделение на обучающую и тестовую выборки?


**Ground truth:**
- expected_facts: оценка качества модели на невидимых данных; предотвращение утечки данных
- forbidden_facts: для увеличения точности на обучающей выборке


**Model output:**
> Разделение данных на обучающую и тестовую выборки помогает оценивать точность моделей. В процессе обучения модель корректируется на основе данных из обучающей выборки. Тестовая выборка используется для оценки работы модели на данных, которые она не видела во время обучения, что позволяет избежать переобучения и проверить, насколько хорошо модель обобщает новые данные.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.222
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_003_regularization (chat_response, language: ru, difficulty: medium)

**User input:** Что такое регуляризация и зачем она нужна?


**Ground truth:**
- expected_facts: добавляет штраф за сложность модели; уменьшает переобучение
- forbidden_facts: регуляризация увеличивает разброс модели


**Model output:**
> Регуляризация — это метод в машинном обучении, который помогает предотвратить модели от переобучения, предотвращая чрезмерную сложность модели и улучшая её обобщающую способность. Это особенно полезно при работе с небольшими или не совсем чистыми данными и в случаях, когда количество признаков сильно превышает количество наблюдений. Регуляризация снижает вероятность чрезмерной адаптации модели к обучающим данным, что помогает повысить точность прогнозов на новых, непроверенных данных.   Если хотите понять это своими словами — это как найти баланс между точностью этапов обучения модели и её способностью работать в незнакомых ситуациях.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.132
  - expected_coverage: 0.0 | forbidden_penalty: 0.0

### theory_004_bias_variance (chat_response, language: ru, difficulty: medium)

**User input:** Объясни компромисс между смещением и разбросом.


**Ground truth:**
- expected_facts: смещение связано с упрощающими предположениями модели; разброс связан с чувствительностью к обучающим данным; существует компромисс между смещением и разбросом
- forbidden_facts: смещение и разброс можно минимизировать независимо


**Model output:**
> Смещение и разброс — это два ключевых понятия в статистике. Смещение — это систематическая ошибка в оценке параметра. Ошибка в оценке параметра считается смещенной, если ее математическое ожидание не равно оцениваемому параметру. Разброс — это мера рассеяния оценок относительно среднего значения. Большое смещение говорит о систематической ошибке, а большой разброс о дисперсиях.  Компромисс между смещением и разбросом исходит из того, что чем меньше смещение, тем больше требуется данных, чтобы снизить разброс и наоборот. Понижение смещения приводит к увеличению объема выборки, что помогает снизить разброс. Малое смещение и узкий разброс — это показатель более надежного предсказания.


**Метрики:**
- decision_score: True
- semantic_similarity: 0.284
  - expected_coverage: 0.0 | forbidden_penalty: 0.0
