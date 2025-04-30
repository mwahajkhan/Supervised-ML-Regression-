Readme

## Folder Structure:

Assignment 4 > Folders (1. Code space 2. Data 3. Presentation, 4. Visualizations)

* Codespace contains: 1. Notebook folder with spy notebook, 2. Scripts Folder (with two scripts, one for each feature selection method i.e., f-test, ref)
* Data folder contains: two data files, cleaned\_transactions, online\_transaction\_data
* Presentation folder: contains ppt for the project
* Visualization folder: contains confusion matrixes

## Notebook Description:

This notebook addresses the problem of detecting unusual (potentially fraudulent) purchase transactions. The dataset contains imbalanced classes, which is a common challenge in fraud detection.

## Notebook Structure:

The notebook is divided into the following sections:

* Task 1: KPIs
* Libraries
* Data Target Class Distribution
* Project\_Script\_1 (Statistical Test: ANOVA F-test(f\_classifier), Feature Selection: F-statistic)
* Project\_Script\_2 (Feature Selection: Recursive Feature Elimination)
* Task 2-7 (Divided work distribution)
* Ethical Concerns

## Data Files:

The analysis uses the following data files:

* Cleaned\_Transactions.csv (Path: content/Cleaned\_Transactions.csv)
* Online\_transaction\_data.csv (Path: content/Online\_transaction\_data.csv)

**Note:** To obtain the "Cleaned\_data" file, run the Project\_Script\_1/Project\_Script\_2 and wait for a few minutes for the file to be downloaded. Alternatively, the "Cleaned\_data" file is provided in the project folder.

**Important:** Before running the script, please update the file paths in the data loading section to match the location of these files on your system or within the Colab environment.

## Models:

The following machine learning models are instantiated:

* Decision Tree
* Logistic Regression
* Naïve Bayes

## Feature Selection & Statistical Tests:

This project employs two distinct feature selection methodologies to identify the most relevant predictors for transaction type:

* **Project\_Script\_1: ANOVA F-test**
    * Statistical Test: ANOVA F-test (f\_classif)
    * Feature Selection: F-statistic
    * This method is a filter approach that evaluates the statistical significance of each feature's relationship with the target variable (Type\_of\_Transaction). It ranks features based on their F-statistic and selects those with a score greater than 10.
* **Project\_Script\_2: Recursive Feature Elimination (RFE)**
    * Feature Selection: Recursive Feature Elimination (RFE)
    * This method is a wrapper approach that works by recursively fitting a model (in this case, Logistic Regression), removing the least significant features, and then refitting. It selects features by considering the model's performance. The number of features selected was matched to the number selected by the ANOVA F-test for comparison.

## Task 1: KPI Selection (Khan)

For the business problem of detecting unusual (potentially fraudulent) purchase transactions, the following Key Performance Indicators (KPIs) are relevant:

* **Precision:** Measures the proportion of correctly identified fraudulent transactions out of all transactions flagged as fraudulent. High precision is crucial to minimize false positives.
* **Recall (Sensitivity):** Measures the proportion of actual fraudulent transactions that are correctly identified. High recall is important to minimize false negatives.
* **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
* **Area Under the Precision-Recall Curve (AUC-PR):** Useful for imbalanced datasets, it evaluates the trade-off between precision and recall across different classification thresholds.
* **False Positive Rate (FPR):** Measures the proportion of genuine transactions that are incorrectly flagged as fraudulent. Minimizing FPR is important to maintain customer trust.
* **Accuracy:** While commonly used, accuracy can be misleading in imbalanced datasets. It measures the overall proportion of correctly classified instances but may be skewed by the large number of true negatives.

These KPIs are selected because they directly address the core concerns of fraud detection: accurately identifying fraudulent activities while minimizing disruption to legitimate transactions. Precision and Recall are particularly important in this type of Machine Learning Problem.

## Important Libraries

```python
from IPython import get_ipython
from IPython.display import display
import pandas as pd
from google.colab import files
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
