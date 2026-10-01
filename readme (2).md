<!-- To Bring back the link to top -->
<a name="readme-top"></a>

# 🌷 Iris Flower Classifier

[![Python][python-shield]][python-url]
[![Jupyter Notebook][jupyter-shield]][jupyter-url]
[![Pandas][pandas-shield]][pandas-url]
[![NumPy][numpy-shield]][numpy-url]
[![Scikit Learn][sklearn-shield]][sklearn-url]
[![Matplotlib][matplotlib-shield]][matplotlib-url]
[![Seaborn][seaborn-shield]][seaborn-url]

<!-- MARKDOWN LINKS & IMAGES -->

[python-shield]: https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[jupyter-shield]: https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/
[pandas-shield]: https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[numpy-shield]: https://img.shields.io/badge/NumPy-Data%20Processing-blue?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[sklearn-shield]: https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[matplotlib-shield]: https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge
[matplotlib-url]: https://matplotlib.org/
[seaborn-shield]: https://img.shields.io/badge/Seaborn-Visualization-teal?style=for-the-badge
[seaborn-url]: https://seaborn.pydata.org/

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h1 align="center">🌷 Iris Flower Classifier</h1>

  <p align="center">
    A machine learning classification project for predicting Iris flower species
    using multiple supervised learning algorithms.
    <br />
    <br />
    <a href="#about-the-project-">Explore the Project</a>
    ·
    <a href="#model-evaluation-">View Results</a>
    ·
    <a href="#getting-started-">Get Started</a>
  </p>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project-">About The Project</a>
      <ul>
        <li><a href="#project-objectives-">Project Objectives</a></li>
        <li><a href="#project-workflow-">Project Workflow</a></li>
        <li><a href="#dataset-">Dataset</a></li>
      </ul>
    </li>
    <li><a href="#exploratory-data-analysis-">Exploratory Data Analysis</a></li>
    <li><a href="#machine-learning-models-">Machine Learning Models</a></li>
    <li><a href="#model-evaluation-">Model Evaluation</a></li>
    <li><a href="#hyperparameter-tuning-">Hyperparameter Tuning</a></li>
    <li><a href="#sample-prediction-">Sample Prediction</a></li>
    <li><a href="#built-with-️">Built With</a></li>
    <li>
      <a href="#getting-started-">Getting Started</a>
      <ul>
        <li><a href="#prerequisites-">Prerequisites</a></li>
        <li><a href="#installation-">Installation</a></li>
      </ul>
    </li>
    <li><a href="#project-structure-">Project Structure</a></li>
    <li><a href="#key-learnings-">Key Learnings</a></li>
    <li><a href="#future-improvements-">Future Improvements</a></li>
    <li><a href="#license-">License</a></li>
    <li><a href="#contact-️">Contact</a></li>
  </ol>
</details>

<!-- About the project -->
## About the Project 💻

The **Iris Flower Classifier** is a beginner-friendly machine learning project
that predicts the species of an Iris flower from four measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project follows a complete machine learning workflow, starting with data
preprocessing and exploratory data analysis and continuing through model
training, evaluation, comparison, and hyperparameter tuning.

The target classes are:

- 🌱 Iris-setosa
- 🌸 Iris-versicolor
- 🌺 Iris-virginica

---

## Project Objectives 🎯

The main objectives of this project are:

1. Understand and preprocess a real-world classification dataset.
2. Explore relationships between flower measurements using visualizations.
3. Train different supervised machine learning classification models.
4. Compare model performance using evaluation metrics.
5. Apply hyperparameter tuning using GridSearchCV.
6. Predict the species of a new Iris flower from its measurements.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Project Workflow -->
## Project Workflow 📚

The project follows a structured machine learning workflow:

**1) Data Loading:**  
Load the `Iris.csv` dataset using Pandas.

**2) Data Preprocessing:**  
Remove the `Id` column, check missing values, remove duplicates, and prepare the dataset for modeling.

**3) Feature Selection:**  
Use the four flower measurements as input features and `Species` as the target variable.

**4) Exploratory Data Analysis:**  
Analyze species distribution and relationships between sepal and petal measurements using plots.

**5) Train-Test Split:**  
Split the dataset into 80% training data and 20% testing data using `random_state=42`.

**6) Model Training:**  
Train Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers.

**7) Model Evaluation:**  
Evaluate the models using accuracy, confusion matrix, precision, recall, and F1-score.

**8) Hyperparameter Tuning:**  
Use GridSearchCV with 5-fold cross-validation to tune KNN and Decision Tree models.

**9) Prediction:**  
Use the trained classifier to predict the species of a new flower.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Dataset -->
## Dataset 🌷

The project uses the **Iris dataset** stored in `Iris.csv`.

| Property | Details |
|---|---|
| Total records | 150 |
| Input features | 4 |
| Target | Species |
| Classes | 3 |
| Records per class | 50 |
| ID column | Removed before training |

### Input Features

| Feature | Description |
|---|---|
| `SepalLengthCm` | Sepal length |
| `SepalWidthCm` | Sepal width |
| `PetalLengthCm` | Petal length |
| `PetalWidthCm` | Petal width |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EDA -->
## Exploratory Data Analysis 🔍

Several visualizations were created to understand the dataset:

- Species count bar chart
- Sepal length vs. sepal width scatter plot
- Petal length vs. petal width scatter plot
- Pair plot
- Correlation heatmap
- Box plots
- Feature relationship scatter plots

### Key Observation

The petal measurements provide clearer separation between Iris species than
the sepal measurements. Petal length and petal width show strong relationships
with species classification.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Models -->
## Machine Learning Models 🤖

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model. It learns
relationships between the input flower measurements and the three species classes.

### 2. K-Nearest Neighbors

KNN classifies a new flower by comparing it with nearby training samples.
The initial model uses:

```text
n_neighbors = 5
```

### 3. Decision Tree

Decision Tree creates a series of decision rules based on the flower measurements
to classify the Iris species.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Evaluation -->
## Model Evaluation 📊

The models were evaluated using an 80/20 train-test split.

| Model | Configuration | Test Accuracy |
|---|---|---:|
| Logistic Regression | `max_iter=200` | **93.33%** |
| KNN | `n_neighbors=5` | **93.33%** |
| Decision Tree | `random_state=42` | **93.33%** |

The baseline models achieved the same test accuracy on the selected test split.

### Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Hyperparameter Tuning -->
## Hyperparameter Tuning ⚙️

GridSearchCV with 5-fold cross-validation was used to tune the KNN and
Decision Tree models.

### Tuned KNN

```text
n_neighbors = 11
weights = uniform
metric = euclidean
```

- Cross-validation accuracy: **99.13%**
- Held-out test accuracy: **90.00%**

### Tuned Decision Tree

```text
criterion = gini
max_depth = 4
min_samples_split = 2
min_samples_leaf = 1
```

- Cross-validation accuracy: **94.89%**
- Held-out test accuracy: **93.33%**

### Important Observation

A higher cross-validation score did not necessarily produce a higher
held-out test accuracy. This demonstrates why model performance should be
checked on unseen test data after tuning.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Sample Prediction -->
## Sample Prediction 🌸

A sample flower with the following measurements was tested:

```text
Sepal Length = 5.1
Sepal Width  = 3.5
Petal Length = 1.4
Petal Width  = 0.2
```

The Logistic Regression model predicts:

```text
Iris-setosa
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Built with -->
## Built With 🖥️

- 🐍 Python
- 📓 Jupyter Notebook
- 🐼 Pandas
- 🔢 NumPy
- 📊 Matplotlib
- 📈 Seaborn
- 🤖 Scikit-learn
- 💻 VS Code / Jupyter

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Getting Started -->
## Getting Started 🚀

Follow these steps to run the project locally.

### Prerequisites 📋

Make sure Python is installed on your computer.

You can download Python from:

https://www.python.org/downloads/

### Installation 📋

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

2. Move into the project directory:

```bash
cd YOUR-REPOSITORY
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Make sure `Iris.csv` is available in the project folder.

5. Run the Python files or open the Jupyter Notebook.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Project Structure -->
## Project Structure 📁

```text
Iris-Flower-Classifier/
│
├── Iris.csv
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Iris_Classifier.ipynb
│
├── src/
│   └── iris_classifier.py
│
└── images/
    ├── species_distribution.png
    ├── pairplot.png
    ├── correlation_heatmap.png
    └── confusion_matrix.png
```

> The folder structure can be adjusted to match the files you actually upload
> to your repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Key Learnings -->
## Key Learnings 📚

Through this project, I learned:

- How to clean and prepare a dataset for machine learning.
- How to perform exploratory data analysis.
- How to identify useful features using visualization.
- How Logistic Regression, KNN, and Decision Tree work.
- How to evaluate classification models.
- How to use confusion matrices and classification reports.
- How to perform hyperparameter tuning with GridSearchCV.
- How cross-validation and test-set evaluation can produce different results.
- How to organize a machine learning project for GitHub.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Future Improvements -->
## Future Improvements 🔮

Possible future improvements include:

- Build a simple web interface for Iris prediction.
- Deploy the classifier as a web application.
- Save and load the trained model using Joblib.
- Add more machine learning algorithms.
- Add interactive visualizations.
- Create an automated prediction API.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- License -->
## License 📄

This project is created for educational and learning purposes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Contact -->
## Contact ☎️

**Kiranmai**

If you have questions or suggestions about this project, you can open an
Issue in the GitHub repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<div align="center">

### 🌷 Thank You for Visiting My Project! ⭐

</div>
