import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

 
# 2. CLEANING DATA
 
df = df.drop("Id", axis=1)

df.dropna(inplace=True)

print("\nNull values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print("\nDataset after removing duplicates:")
print(df)
 
# 3. UNIQUE VALUES AND COUNTS

print("\nUnique Species:")
print(df["Species"].unique())

print("\nSpecies Counts:")
print(df["Species"].value_counts())

# 4. MATPLOTLIB - SPECIES COUNT

plt.figure(figsize=(7, 5))

species_counts = df["Species"].value_counts()

plt.bar(
    species_counts.index,
    species_counts.values
)

plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.xticks(rotation=15)

plt.show()

# 5. FEATURE SELECTION
 
x = df.drop("Species", axis=1)

y = df["Species"]

print("\nFeatures:")
print(x.head())

print("\nLabels:")
print(y.head())

# 6. MATPLOTLIB - SEPAL LENGTH VS SEPAL WIDTH
 
plt.figure(figsize=(8, 6))

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    plt.scatter(
        data["SepalLengthCm"],
        data["SepalWidthCm"],
        label=species
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()

plt.show()
 
# 7. MATPLOTLIB - PETAL LENGTH VS PETAL WIDTH
 
plt.figure(figsize=(8, 6))

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    plt.scatter(
        data["PetalLengthCm"],
        data["PetalWidthCm"],
        label=species
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()
 
# 8. WEEK 4 - PAIR PLOT
 
print("\nCreating Pair Plot...")

sns.pairplot(
    df,
    hue="Species",
    diag_kind="hist"
)

plt.suptitle(
    "Pair Plot - Iris Feature Relationships",
    y=1.02
)

plt.show()

# 9. WEEK 4 - CORRELATION HEATMAP
 
print("\nCreating Correlation Heatmap...")

numeric_features = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

correlation = numeric_features.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Iris Features")

plt.show()

# 10. WEEK 4 - BOX PLOT: SEPAL LENGTH

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="SepalLengthCm"
)

plt.title("Sepal Length Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")

plt.show()

# 11. WEEK 4 - BOX PLOT: SEPAL WIDTH

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="SepalWidthCm"
)

plt.title("Sepal Width Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")

plt.show()

# 12. WEEK 4 - BOX PLOT: PETAL LENGTH


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="PetalLengthCm"
)

plt.title("Petal Length Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")

plt.show()

# 13. WEEK 4 - BOX PLOT: PETAL WIDTh
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="PetalWidthCm"
)

plt.title("Petal Width Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Width (cm)")

plt.show()
# 14. WEEK 4 - SCATTER PLOT

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species"
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()

plt.show()

# 15. WEEK 4 - SCATTER PLOT

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalWidthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Sepal Width vs Petal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()

# 16. WEEK 4 - SCATTER PLOT

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Sepal Length vs Petal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()
# 17. WEEK 4 - SCATTER PLOT

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalWidthCm",
    y="PetalLengthCm",
    hue="Species"
)

plt.title("Sepal Width vs Petal Length")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()

plt.show()

# 18. WEEK 4 - FEATURE ANALYSIS
 
print("\n========================================")
print("WEEK 4 - FEATURE ANALYSIS")
print("========================================")

print("""
1. Petal Length and Petal Width show a strong positive
   correlation.

2. The Pair Plot shows that Petal Length and Petal Width
   provide better separation between the three Iris species.

3. The Box Plots show clear differences in petal measurements
   among the three species.

4. Sepal Length and Sepal Width show more overlap between
   the species.

5. Petal Length and Petal Width are the most useful features
   for distinguishing Iris species.
""")

# 19. WEEK 4 - CONCLUSION

print("\n========================================")
print("WEEK 4 - CONCLUSION")
print("========================================")

print("""
Data visualization was performed using Pair Plots,
Correlation Heatmap, Box Plots, and Scatter Plots.

The visualizations show that Petal Length and Petal Width
have a strong relationship and provide better separation
between the three Iris species.

Sepal Length and Sepal Width have more overlap between
the species.

Therefore, Petal Length and Petal Width are the most
important features for identifying the Iris species.
""")
# 20. TRAINING AND TESTING DATA

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", x_train.shape)
print("Testing data size:", x_test.shape)

# 21. CREATE LOGISTIC REGRESSION MODEL

model = LogisticRegression(max_iter=200)


model.fit(x_train, y_train)

# 22. PREDICTION

y_pred = model.predict(x_test)

print("\nPredicted values:")
print(y_pred)

# 23. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# 24. CONFUSION MATRIX
 
confusion = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(confusion)

# 25. MATPLOTLIB - CONFUSION MATRIX
 
plt.figure(figsize=(7, 6))

plt.imshow(confusion)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.colorbar()
species_names = model.classes_

plt.xticks(
    range(len(species_names)),
    species_names,
    rotation=15
)

plt.yticks(
    range(len(species_names)),
    species_names
)
 
for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.text(
            j,
            i,
            confusion[i, j],
            ha="center",
            va="center"
        )

plt.show()

# 26. CLASSIFICATION REPORT
 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 27. ACTUAL VS PREDICTED
 
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison)
# 28. MATPLOTLIB - ACTUAL VS PREDICTED

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted"
)

plt.title("Actual vs Predicted Species")
plt.xlabel("Test Sample")
plt.ylabel("Species")

plt.legend()

plt.xticks(range(len(y_test)))

plt.show()

# 29. PREDICT A NEW FLOWER

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=x.columns
)

prediction = model.predict(new_flower)

print("\nNew Flower Details:")
print(new_flower)

print("\nPredicted Species:", prediction[0])
# 30. FINAL PROJECT SUMMARY

print("\n========================================")
print("FINAL PROJECT SUMMARY")
print("========================================")

print("""
Dataset: Iris Dataset

Machine Learning Algorithm:
Logistic Regression

Data Visualization:
- Species Count Bar Chart
- Sepal Scatter Plot
- Petal Scatter Plot
- Pair Plot
- Correlation Heatmap
- Box Plots
- Additional Scatter Plots

Important Features:
- Petal Length
- Petal Width

Model Accuracy:
""")

print(accuracy * 100, "%")

print("""
Final Finding:
Petal Length and Petal Width are the most useful features
for distinguishing the three Iris species.
""")


# =========================================================
# WEEK 5 - KNN VS DECISION TREE
# =========================================================

print("\n========================================")
print("WEEK 5 - KNN VS DECISION TREE")
print("========================================")

# 31. IMPORT WEEK 5 LIBRARIES

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 32. TRAIN KNN MODEL

print("\n--- KNN MODEL ---")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_accuracy)
print("KNN Accuracy Percentage:", knn_accuracy * 100, "%")

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))


# 33. KNN CONFUSION MATRIX

knn_confusion = confusion_matrix(y_test, knn_pred)

print("\nKNN Confusion Matrix:")
print(knn_confusion)

plt.figure(figsize=(7, 6))
plt.imshow(knn_confusion)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.colorbar()

species_names = knn.classes_

plt.xticks(
    range(len(species_names)),
    species_names,
    rotation=15
)

plt.yticks(
    range(len(species_names)),
    species_names
)

for i in range(len(knn_confusion)):
    for j in range(len(knn_confusion)):
        plt.text(
            j,
            i,
            knn_confusion[i, j],
            ha="center",
            va="center"
        )

plt.show()


# 34. TRAIN DECISION TREE MODEL

print("\n--- DECISION TREE MODEL ---")

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(x_train, y_train)

dt_pred = decision_tree.predict(x_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Accuracy Percentage:", dt_accuracy * 100, "%")

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))


# 35. DECISION TREE CONFUSION MATRIX

dt_confusion = confusion_matrix(y_test, dt_pred)

print("\nDecision Tree Confusion Matrix:")
print(dt_confusion)

plt.figure(figsize=(7, 6))
plt.imshow(dt_confusion)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.colorbar()

species_names = decision_tree.classes_

plt.xticks(
    range(len(species_names)),
    species_names,
    rotation=15
)

plt.yticks(
    range(len(species_names)),
    species_names
)

for i in range(len(dt_confusion)):
    for j in range(len(dt_confusion)):
        plt.text(
            j,
            i,
            dt_confusion[i, j],
            ha="center",
            va="center"
        )

plt.show()


# 36. COMPARE KNN AND DECISION TREE

comparison_results = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Decision Tree"],
    "Accuracy": [accuracy, knn_accuracy, dt_accuracy],
    "Accuracy Percentage": [
        accuracy * 100,
        knn_accuracy * 100,
        dt_accuracy * 100
    ]
})

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(comparison_results)


# 37. MODEL ACCURACY BAR CHART

plt.figure(figsize=(8, 6))

plt.bar(
    comparison_results["Model"],
    comparison_results["Accuracy Percentage"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=15)

plt.show()


# 38. FIND BEST MODEL

best_model_row = comparison_results.loc[
    comparison_results["Accuracy"].idxmax()
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Best Model:", best_model_row["Model"])
print("Best Accuracy:", best_model_row["Accuracy Percentage"], "%")


# 39. WEEK 5 CONCLUSION

print("""
========================================
WEEK 5 - CONCLUSION
========================================

KNN and Decision Tree models were trained
using the same Iris training and testing data.

KNN predicts the species by checking the
nearest data points and using majority voting.

Decision Tree predicts the species by creating
decision rules based on the input features.

Both models were evaluated using accuracy,
classification report, and confusion matrix.

The accuracies of Logistic Regression, KNN,
and Decision Tree were compared.

The model with the highest accuracy is selected
as the best-performing model for this experiment.
""")


# =========================================================
# WEEK 6 - HYPERPARAMETER TUNING
# =========================================================

print("\n========================================")
print("WEEK 6 - HYPERPARAMETER TUNING")
print("========================================")

# 40. IMPORT HYPERPARAMETER TUNING LIBRARIES

from sklearn.model_selection import GridSearchCV


# 41. KNN HYPERPARAMETER TUNING

print("\n--- KNN HYPERPARAMETER TUNING ---")

knn_param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=5,
    scoring="accuracy"
)

knn_grid.fit(x_train, y_train)

print("Best KNN Parameters:")
print(knn_grid.best_params_)

print("Best KNN Cross-Validation Accuracy:")
print(knn_grid.best_score_)

best_knn = knn_grid.best_estimator_

best_knn_pred = best_knn.predict(x_test)

best_knn_accuracy = accuracy_score(y_test, best_knn_pred)

print("Tuned KNN Test Accuracy:", best_knn_accuracy)
print("Tuned KNN Test Accuracy Percentage:",
      best_knn_accuracy * 100, "%")


# 42. DECISION TREE HYPERPARAMETER TUNING

print("\n--- DECISION TREE HYPERPARAMETER TUNING ---")

dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring="accuracy"
)

dt_grid.fit(x_train, y_train)

print("Best Decision Tree Parameters:")
print(dt_grid.best_params_)

print("Best Decision Tree Cross-Validation Accuracy:")
print(dt_grid.best_score_)

best_dt = dt_grid.best_estimator_

best_dt_pred = best_dt.predict(x_test)

best_dt_accuracy = accuracy_score(y_test, best_dt_pred)

print("Tuned Decision Tree Test Accuracy:",
      best_dt_accuracy)

print("Tuned Decision Tree Test Accuracy Percentage:",
      best_dt_accuracy * 100, "%")


# 43. COMPARE BEFORE AND AFTER TUNING

tuning_results = pd.DataFrame({
    "Model": [
        "KNN Before Tuning",
        "KNN After Tuning",
        "Decision Tree Before Tuning",
        "Decision Tree After Tuning"
    ],
    "Accuracy": [
        knn_accuracy,
        best_knn_accuracy,
        dt_accuracy,
        best_dt_accuracy
    ],
    "Accuracy Percentage": [
        knn_accuracy * 100,
        best_knn_accuracy * 100,
        dt_accuracy * 100,
        best_dt_accuracy * 100
    ]
})

print("\n========================================")
print("BEFORE AND AFTER TUNING")
print("========================================")

print(tuning_results)


# 44. ACCURACY COMPARISON GRAPH

plt.figure(figsize=(10, 6))

plt.bar(
    tuning_results["Model"],
    tuning_results["Accuracy Percentage"]
)

plt.title("Accuracy Before and After Hyperparameter Tuning")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=20)

plt.show()


# 45. TUNED KNN CONFUSION MATRIX

best_knn_confusion = confusion_matrix(
    y_test,
    best_knn_pred
)

print("\nTuned KNN Confusion Matrix:")
print(best_knn_confusion)

plt.figure(figsize=(7, 6))
plt.imshow(best_knn_confusion)

plt.title("Tuned KNN Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.colorbar()

species_names = best_knn.classes_

plt.xticks(
    range(len(species_names)),
    species_names,
    rotation=15
)

plt.yticks(
    range(len(species_names)),
    species_names
)

for i in range(len(best_knn_confusion)):
    for j in range(len(best_knn_confusion)):
        plt.text(
            j,
            i,
            best_knn_confusion[i, j],
            ha="center",
            va="center"
        )

plt.show()


# 46. TUNED DECISION TREE CONFUSION MATRIX

best_dt_confusion = confusion_matrix(
    y_test,
    best_dt_pred
)

print("\nTuned Decision Tree Confusion Matrix:")
print(best_dt_confusion)

plt.figure(figsize=(7, 6))
plt.imshow(best_dt_confusion)

plt.title("Tuned Decision Tree Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.colorbar()

species_names = best_dt.classes_

plt.xticks(
    range(len(species_names)),
    species_names,
    rotation=15
)

plt.yticks(
    range(len(species_names)),
    species_names
)

for i in range(len(best_dt_confusion)):
    for j in range(len(best_dt_confusion)):
        plt.text(
            j,
            i,
            best_dt_confusion[i, j],
            ha="center",
            va="center"
        )

plt.show()


# 47. FINAL TUNED MODEL COMPARISON

final_model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Tuned KNN",
        "Tuned Decision Tree"
    ],
    "Accuracy": [
        accuracy,
        best_knn_accuracy,
        best_dt_accuracy
    ],
    "Accuracy Percentage": [
        accuracy * 100,
        best_knn_accuracy * 100,
        best_dt_accuracy * 100
    ]
})

print("\n========================================")
print("FINAL MODEL COMPARISON AFTER TUNING")
print("========================================")

print(final_model_results)


# 48. FIND BEST TUNED MODEL

best_final_row = final_model_results.loc[
    final_model_results["Accuracy"].idxmax()
]

print("\n========================================")
print("BEST FINAL MODEL")
print("========================================")

print("Best Model:", best_final_row["Model"])
print(
    "Best Accuracy:",
    best_final_row["Accuracy Percentage"],
    "%"
)


# 49. WEEK 6 CONCLUSION

print("""
========================================
WEEK 6 - CONCLUSION
========================================

Hyperparameter tuning was performed for
KNN and Decision Tree models using GridSearchCV.

For KNN, different values of K, weights,
and distance metrics were tested.

For Decision Tree, criterion, maximum depth,
minimum samples for splitting, and minimum
samples per leaf were tested.

The best parameters were selected using
5-fold cross-validation.

The tuned models were evaluated on the
test dataset and compared with their
untuned versions.

Hyperparameter tuning helps select better
model settings and can improve model
performance.

The model with the highest final test
accuracy is selected as the best model.
""")
