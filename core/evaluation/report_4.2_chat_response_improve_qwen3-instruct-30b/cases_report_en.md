# Детальный отчёт по кейсам (en)

## Кейсы
### chat_001_what_is_iris_en (chat_response, language: en, difficulty: easy)

**User input:** Explain what this dataset represents.


**Ground truth:**
- expected_facts: dataset contains measurements of petals and sepals; three iris species


**Model output:**
> This dataset contains measurements of flower parts from different species of iris plants.  It has five columns:  - **sepal_length**: The length of the sepal, in centimeters. - **sepal_width**: The width of the sepal, in centimeters. - **petal_length**: The length of the petal, in centimeters. - **petal_width**: The width of the petal, in centimeters. - **species**: The type of iris plant, such as setosa, versicolor, or virginica.  These measurements are used to help identify or classify the species of an iris flower based on its physical features.
- decision_score: True

### chat_002_how_many_classes_en (chat_response, language: en, difficulty: easy)

**User input:** How many classes are in this dataset?


**Ground truth:**
- expected_facts: three classes; setosa; versicolor; virginica


**Model output:**
> The dataset has **three classes**.  These classes are the unique values in the **species** column.  The species are:   - **setosa**   - **versicolor**   - **virginica**
- decision_score: True

### chat_003_best_features_en (chat_response, language: en, difficulty: medium)

**User input:** Which features best separate the iris species?


**Ground truth:**
- expected_facts: petal length; petal width


**Model output:**
> The features that best separate the iris species are **petal length** and **petal width**.  These two features show the clearest differences between the species when looking at the data.  **Sepal length** and **sepal width** have more overlap between species, so they are less effective at separation.  This conclusion comes from observing how the values for each species cluster in the dataset.
- decision_score: True

### chat_004_easiest_class_en (chat_response, language: en, difficulty: medium)

**User input:** Which iris species is the easiest to separate from the others?


**Ground truth:**
- expected_facts: setosa; shorter petals; wider petals


**Model output:**
> Based on the available data, **Iris setosa** is the easiest to separate from the other species.  This is because **Iris setosa** has distinctly smaller petal and sepal measurements compared to **Iris versicolor** and **Iris virginica**.  For example: - **Petal length** and **petal width** for setosa are consistently smaller. - There is little to no overlap in these features between setosa and the other two species.  So, using **petal length** or **petal width** alone, you can usually tell setosa apart with high confidence.  The other two species, versicolor and virginica, have more overlapping measurements, making them harder to separate.  Note: This conclusion is based on general patterns in the dataset and does not require calculations or plots.
- decision_score: True

### theory_001_overfitting_en (chat_response, language: en, difficulty: easy)

**User input:** What is overfitting in machine learning?


**Ground truth:**
- expected_facts: model fits training data too well; poor generalization to new data
- forbidden_facts: overfitting improves performance on unseen data


**Model output:**
> **Overfitting** happens when a machine learning model learns the training data too well — so well that it starts memorizing the noise or random details instead of finding the true patterns.  This means the model performs very well on the training data, but it doesn’t work well on new, unseen data.  For example, if a model is trained to classify flowers using features like **sepal_length**, **sepal_width**, **petal_length**, and **petal_width**, overfitting would occur if the model uses tiny, random variations in the training data to make predictions — instead of learning general rules for each **species**.  Overfitting often happens when a model is too complex, like having too many rules or layers, especially with little training data.  To avoid overfitting, we use techniques like **cross-validation**, **regularization**, or simplifying the model.  In short: **Overfitting = memorizing the training data instead of learning to generalize.**
- decision_score: True

### theory_002_train_test_split_en (chat_response, language: en, difficulty: easy)

**User input:** Why is a train-test split used?


**Ground truth:**
- expected_facts: evaluate model on unseen data; prevent data leakage
- forbidden_facts: to increase training accuracy


**Model output:**
> A train-test split is used to check how well a machine learning model will work on new, unseen data.  The data is divided into two parts:  - **Training data**: The model learns from this part. - **Test data**: The model is tested on this part to see how well it performs.  This helps us avoid **overfitting**, which happens when a model memorizes the training data but doesn’t work well on new examples.  Using a train-test split gives a more honest estimate of the model’s real-world performance.
- decision_score: True

### theory_003_regularization_en (chat_response, language: en, difficulty: medium)

**User input:** What is regularization and why is it used?


**Ground truth:**
- expected_facts: adds penalty to model complexity; reduces overfitting
- forbidden_facts: regularization increases model variance


**Model output:**
> **Regularization** is a technique used in machine learning to prevent a model from becoming too complex.  It works by adding a small penalty to the size of the model's coefficients, especially when the model starts to fit the training data too closely.  This helps reduce **overfitting**, which happens when a model learns the training data too well, including its noise and outliers, and then performs poorly on new, unseen data.  Regularization makes the model simpler and more likely to generalize well to new examples.  Common types include **L1 regularization** and **L2 regularization**, which add different kinds of penalties to the model's weights.  It is used to improve a model's performance on data it hasn’t seen before.
- decision_score: True

### theory_004_bias_variance_en (chat_response, language: en, difficulty: medium)

**User input:** Explain the bias-variance tradeoff.


**Ground truth:**
- expected_facts: bias refers to error from simplifying assumptions; variance refers to sensitivity to training data; tradeoff between bias and variance
- forbidden_facts: bias and variance can both be minimized independently


**Model output:**
> The **bias-variance tradeoff** is a key idea in machine learning that helps us understand how well a model will perform on new data.  **Bias** is how far off the model's predictions are from the true values, on average. High bias means the model is too simple and doesn’t capture the patterns in the data. It might consistently make the same kind of mistake.  **Variance** is how much the model’s predictions change when trained on different data. High variance means the model is too sensitive to small changes in the training data. It might fit the training data very well but do poorly on new data.  A good model has **low bias** and **low variance**. But there’s a tradeoff: reducing bias often increases variance, and reducing variance often increases bias.  For example, a very simple model like a straight line might have high bias (it can’t follow curved patterns) but low variance (it doesn’t change much with new data). A very complex model, like a high-degree polynomial, might have low bias (it fits the training data closely) but high variance (it overfits and fails on new data).  So, the goal is to find a balance — a model that generalizes well to new data by minimizing both bias and variance.
- decision_score: True

### chat_005_feature_count_en (chat_response, language: en, difficulty: easy)

**User input:** How many features does this dataset have?


**Ground truth:**
- expected_facts: four features; sepal length; sepal width; petal length; petal width
- forbidden_facts: five features; only petal features


**Model output:**
> This dataset has **four features**.  They are: - **sepal_length** - **sepal_width** - **petal_length** - **petal_width**  These are all numerical measurements. The **species** column is a label or target, not a feature.
- decision_score: True

### chat_006_samples_count_en (chat_response, language: en, difficulty: easy)

**User input:** How many samples are in this dataset?


**Ground truth:**
- expected_facts: 150 samples; 50 samples per class
- forbidden_facts: 100 samples; imbalanced classes


**Model output:**
> I don’t have enough information to be certain how many samples are in this dataset.
- decision_score: True

### chat_007_numeric_only_en (chat_response, language: en, difficulty: easy)

**User input:** Are all features in this dataset numeric?


**Ground truth:**
- expected_facts: all features are numeric; continuous measurements
- forbidden_facts: categorical features; text features


**Model output:**
> No, not all features in this dataset are numeric.  The features **sepal_length**, **sepal_width**, **petal_length**, and **petal_width** are all numeric (specifically, float values).  However, the feature **species** is a category, meaning it contains text labels for different species, not numbers.  So, the dataset has **four numeric features** and **one categorical feature**.
- decision_score: True

### chat_008_need_scaling_en (chat_response, language: en, difficulty: medium)

**User input:** Is feature scaling important for this dataset?


**Ground truth:**
- expected_facts: important for distance-based models; helps algorithms like kNN or SVM
- forbidden_facts: scaling is never useful; scaling changes class labels


**Model output:**
> Feature scaling can be important for this dataset, especially if you're using machine learning models that are sensitive to the scale of input features.  For example, models like **k-nearest neighbors**, **support vector machines**, or **neural networks** work better when all features are on a similar scale. In this dataset, the features — **sepal_length**, **sepal_width**, **petal_length**, and **petal_width** — are measured in centimeters, but their ranges may differ.  If one feature has much larger values than others, it could dominate the distance calculations or gradient updates, even if it’s not more important.  However, some models, like **decision trees** or **random forests**, are not affected by feature scaling because they make decisions based on feature order, not magnitude.  So, whether scaling is important depends on the model you plan to use.  If you're unsure, it's often safe to scale the features when working with this dataset.
- decision_score: True

### chat_009_linear_separable_en (chat_response, language: en, difficulty: medium)

**User input:** Is this dataset linearly separable?


**Ground truth:**
- expected_facts: setosa is linearly separable; other classes overlap
- forbidden_facts: all classes are perfectly separable; no class is separable


**Model output:**
> I don't have enough information to be certain if this dataset is linearly separable.  To determine linear separability, we would need to examine how the **species** classes are distributed in relation to the features like **sepal_length**, **sepal_width**, **petal_length**, and **petal_width**.  Linear separability means that a straight line (in 2D), plane (in 3D), or hyperplane (in higher dimensions) can separate the classes without overlap.  Without visualizing the data or running a separation test, we cannot confirm this.  If you'd like, I can suggest how to check this using Python code.
- decision_score: True

### chat_011_unsupervised_possible_en (chat_response, language: en, difficulty: medium)

**User input:** Can this dataset be used for unsupervised learning?


**Ground truth:**
- expected_facts: can be used for clustering; labels can be ignored
- forbidden_facts: unsupervised learning requires labels


**Model output:**
> Yes, this dataset can be used for unsupervised learning.  Unsupervised learning works with data that does not have labeled outcomes. In this dataset, the **species** column is a label, but we can choose to ignore it.  If we use only the numerical features — **sepal_length**, **sepal_width**, **petal_length**, and **petal_width** — we can apply unsupervised techniques like:  - Clustering to find natural groups in the data. - Dimensionality reduction to simplify the data.  These methods do not need the **species** label to work.  So, yes — this dataset supports unsupervised learning when the **species** label is not used.
- decision_score: True

### chat_012_pca_purpose_en (chat_response, language: en, difficulty: medium)

**User input:** Is PCA applicable to this dataset?


**Ground truth:**
- expected_facts: reduce dimensionality; enable visualization; capture most variance
- forbidden_facts: PCA improves class labels


**Model output:**
> **Yes**, Principal Component Analysis, or **PCA**, is applicable to this dataset.  PCA works on numerical features, and this dataset has four numerical features:   - **sepal_length**   - **sepal_width**   - **petal_length**   - **petal_width**  These are all floating-point numbers, which is exactly what PCA needs.  PCA helps reduce the number of features by finding new directions—called principal components—that capture the most variation in the data.  So, since all the features are numeric and suitable for scaling, PCA can be used to explore patterns or simplify the dataset.  Note: PCA does not use the **species** column. It only works on the numerical measurements.
- decision_score: True

### chat_013_outliers_en (chat_response, language: en, difficulty: medium)

**User input:** Are outliers a major concern in this dataset?


**Ground truth:**
- expected_facts: dataset is relatively clean; few or no strong outliers
- forbidden_facts: dataset is heavily corrupted by outliers


**Model output:**
> Outliers can be a concern in any dataset, but with the information provided, we cannot say for sure whether outliers are a major issue in this dataset.  The dataset includes **sepal_length**, **sepal_width**, **petal_length**, and **petal_width**, all measured in continuous units, and **species** as a categorical label.  To determine if outliers are a major concern, we would need to examine the actual values — for example, by looking at distributions, using visual tools like box plots, or applying statistical methods.  Without seeing the data or specific values, we cannot confirm whether any observations are extreme or unusual.  So, **I don’t have enough information to be certain** if outliers are a major concern.
- decision_score: True

### chat_014_real_world_limitations_en (chat_response, language: en, difficulty: hard)

**User input:** What are the limitations of this dataset?


**Ground truth:**
- expected_facts: small size; not representative of complex real-world data
- forbidden_facts: suitable for deep learning at scale


**Model output:**
> This dataset has a few limitations.  - It only includes **four features**: *sepal_length*, *sepal_width*, *petal_length*, and *petal_width*. These are measurements in centimeters, but no other information like location, time, or environmental conditions is included.  - The target variable is *species*, which has three categories: *setosa*, *versicolor*, and *virginica*. While this is useful for classification, it does not include any other labels or outcomes.  - The dataset is small and well-known, often used for teaching. It may not represent real-world complexity or variation.  - There is no information about how the data was collected, such as sample size, measurement methods, or potential errors.  - Since the dataset is fixed, it cannot be expanded with new features or samples without external data.  In short, this dataset is simple and clean, but limited in scope and real-world detail.
- decision_score: True

### theory_006_classification_vs_regression_en (chat_response, language: en, difficulty: easy)

**User input:** What is the difference between classification and regression?


**Ground truth:**
- expected_facts: classification predicts discrete labels; regression predicts continuous values
- forbidden_facts: classification predicts numbers only; regression predicts categories


**Model output:**
> **Classification** and **regression** are two types of supervised machine learning tasks.  In **classification**, the goal is to predict a **category** or **label**.   For example, deciding if a flower is *setosa*, *versicolor*, or *virginica* based on its measurements.  In **regression**, the goal is to predict a **number** or **continuous value**.   For example, predicting the exact *petal length* of a flower.  So, the main difference is:   - **Classification** predicts **which group** something belongs to.   - **Regression** predicts **how much** of something there is.  Both use data to learn patterns, but they predict different kinds of outcomes.
- decision_score: True

### theory_007_cross_validation_en (chat_response, language: en, difficulty: medium)

**User input:** What is cross-validation?


**Ground truth:**
- expected_facts: splits data into multiple folds; provides more reliable performance estimates; reduces dependence on a single train-test split
- forbidden_facts: cross-validation guarantees perfect performance


**Model output:**
> **Cross-validation** is a method used to check how well a machine learning model will work on new, unseen data.  It works by splitting the data into parts. The model is trained on some parts and tested on the remaining part. This process is repeated several times, with different parts used for testing each time.  The results from all the tests are combined to give a better estimate of the model’s performance.  For example, in **k-fold cross-validation**, the data is divided into k groups. The model is trained on k minus one groups and tested on the one left out. This is done k times, once for each group.  This helps avoid mistakes from a single random split of the data.  Cross-validation is useful when you have limited data and want to make sure your model is reliable.
- decision_score: True

### theory_008_feature_engineering_en (chat_response, language: en, difficulty: medium)

**User input:** What is feature engineering?


**Ground truth:**
- expected_facts: process of creating or transforming features; can improve model performance; requires domain knowledge
- forbidden_facts: feature engineering is only data cleaning


**Model output:**
> **Feature engineering** is the process of using your data to create new features or improve existing ones to help a machine learning model learn better.  For example, with the iris dataset, you might combine **sepal_length** and **sepal_width** to create a new feature like "sepal area" by multiplying them.  It’s about choosing or building features that make patterns in the data easier to see for the model.  You can only use the columns: **sepal_length**, **sepal_width**, **petal_length**, **petal_width**, and **species**.  No other features or values can be invented.
- decision_score: True

### theory_009_data_leakage_en (chat_response, language: en, difficulty: medium)

**User input:** What is data leakage?


**Ground truth:**
- expected_facts: information from test data leaks into training; leads to overly optimistic evaluation; harms generalization
- forbidden_facts: data leakage improves real-world performance


**Model output:**
> **Data leakage** happens when information from outside the training data is used to create a model.  This can make the model seem more accurate than it really is.  For example, if a model uses future data or data that wouldn’t be available when making real predictions, that’s data leakage.  In the context of this dataset, data leakage would occur if we accidentally used values from the *species* column when predicting *species*, or if we used information from the test set during training.  Always make sure the model only uses data that would be available at prediction time.
- decision_score: True

### theory_010_precision_recall_en (chat_response, language: en, difficulty: medium)

**User input:** Explain precision and recall.


**Ground truth:**
- expected_facts: precision measures correctness of positive predictions; recall measures coverage of actual positives; tradeoff between precision and recall
- forbidden_facts: precision and recall are always equal


**Model output:**
> **Precision** and **recall** are two important measures used to evaluate how well a model classifies data, especially in classification tasks.  Let’s say we are trying to identify a specific type of flower, like *setosa*, from a group of flowers.  **Precision** asks: *Out of all the flowers the model said were setosa, how many were actually setosa?*  - High precision means the model makes few mistakes when it predicts a class. - It focuses on **correctness** of positive predictions.  **Recall** asks: *Out of all the actual setosa flowers, how many did the model correctly identify?*  - High recall means the model finds most of the actual positive cases. - It focuses on **completeness** of detection.  For example: - If a model says 10 flowers are setosa, but only 7 are truly setosa, precision is low. - If there are 10 actual setosa flowers, but the model only finds 3, recall is low.  In short: - **Precision** = True Positives / (True Positives + False Positives) - **Recall** = True Positives / (True Positives + False Negatives)  Both are important, but you might care more about one depending on the situation.  For instance: - In medical diagnosis, high recall is often more important — you want to catch all cases. - In spam detection, high precision may be better — you don’t want real emails marked as spam.  These values depend on the data and the model. To calculate them, we need actual predictions and true labels.
- decision_score: True

### theory_011_accuracy_limitations_en (chat_response, language: en, difficulty: medium)

**User input:** Is accuracy a universal metric?


**Ground truth:**
- expected_facts: problematic with imbalanced datasets; does not reflect class-wise performance
- forbidden_facts: accuracy is always the best metric


**Model output:**
> **Accuracy** is a common metric used to evaluate how often a model makes correct predictions.  It is **not** a universal metric because it can be misleading in certain situations.  For example: - If one class is much more common than others, a model might seem accurate just by predicting the majority class. - Accuracy does not consider **false positives** or **false negatives** separately.  So, while accuracy is useful in balanced datasets, it is not always the best choice.  In cases with **imbalanced classes**, other metrics like **precision**, **recall**, or **F1-score** may be more informative.  Always consider the context and data distribution when choosing a metric.
- decision_score: True

### chat_015_what_is_in_data_en (chat_response, language: en, difficulty: easy)

**User input:** What is in the data?


**Ground truth:**
- expected_facts: The data represents the Iris dataset; The dataset contains numeric features describing sepal and petal measurements; There is a categorical feature indicating the iris species; Each row corresponds to a single flower
- forbidden_facts: Mentioning specific statistics or numeric values; Claims about model performance or training results; Assumptions about downstream data processing


**Model output:**
> The data contains five columns:  - **sepal_length**: A measurement in centimeters, representing the length of the sepal. - **sepal_width**: A measurement in centimeters, representing the width of the sepal. - **petal_length**: A measurement in centimeters, representing the length of the petal. - **petal_width**: A measurement in centimeters, representing the width of the petal. - **species**: A category that identifies the type of iris flower. The possible values are: setosa, versicolor, and virginica.  All measurements are numerical, and the species is a label that groups the flowers into three types.
- decision_score: True
