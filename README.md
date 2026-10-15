# Student Performance Prediction Using Logistic Regression

## 1. Introduction

This project applies predictive analytics to student performance data using logistic regression. The model classifies students into pass and fail categories based on academic, demographic, and behavioral attributes.

## 2. Objective

The objectives of this project are to:
- Explore and understand student performance data.
- Perform data preprocessing and exploratory data analysis (EDA).
- Develop a logistic regression classification model.
- Evaluate the model using appropriate performance metrics.

## 3. Dataset

The project uses the Student Performance dataset from the UCI Machine Learning Repository.

**Dataset source:** https://archive.ics.uci.edu/dataset/320/student+performance

The mathematics-course dataset (`student-mat.csv`) was used.

- Total records: 395 students
- Total columns in the original dataset: 33
- Missing values: 0

## 4. Target Variable

The final grade (`G3`) was converted into a binary target variable:

- **Pass (1):** Final grade G3 is greater than or equal to 10 out of 20.
- **Fail (0):** Final grade G3 is below 10.

The `G3` column was excluded from the input features because it directly determines the target variable.

The dataset contains:
- Pass: 265 students
- Fail: 130 students

## 5. Methodology

The project follows these steps:

1. Load and inspect the dataset using pandas.
2. Check missing values.
3. Perform exploratory data analysis using visualizations.
4. Create the binary pass/fail target.
5. Separate the input features and target variable.
6. Impute missing values using median imputation for numerical features and most-frequent imputation for categorical features.
7. Standardize numerical features using StandardScaler.
8. Encode categorical features using OneHotEncoder.
9. Split the dataset into 80% training data and 20% testing data.
10. Train a logistic regression classifier.
11. Evaluate the model using accuracy, F1-score, ROC-AUC, a classification report, and a confusion matrix.

The preprocessing steps and classifier were combined into a scikit-learn pipeline.

## 6. Model Evaluation Results

The model was evaluated on 79 test records after training on 316 records.

| Metric | Result |
|---|---:|
| Accuracy | 87.34% |
| F1-score | 0.9000 |
| ROC-AUC | 0.9485 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Fail | 0.75 | 0.92 | 0.83 | 26 |
| Pass | 0.96 | 0.85 | 0.90 | 53 |
| Macro average | 0.85 | 0.89 | 0.86 | 79 |
| Weighted average | 0.89 | 0.87 | 0.88 | 79 |

### Interpretation

- **Accuracy:** The model correctly classified approximately 87.34% of the test records.
- **F1-score:** The reported F1-score for the Pass class is 0.90, indicating a strong balance between precision and recall for that class.
- **ROC-AUC:** The score of 0.9485 indicates strong ability to distinguish between pass and fail outcomes.
- **Fail classification:** The model identified 24 of the 26 actual failing students correctly, while misclassifying 2 as passing.
- **Pass classification:** The model correctly identified 45 of the 53 actual passing students, while misclassifying 8 as failing.

These results indicate promising performance on this test split, but they do not guarantee similar performance on other student populations.

## 7. Exploratory Data Analysis

Three visualizations were generated and saved in the `outputs` directory.

### 7.1 Student Pass/Fail Distribution

The bar chart shows 265 students classified as passing and 130 classified as failing under the selected grade threshold. This indicates that the dataset contains more passing students than failing students.

### 7.2 Study Time vs Final Grade

The box plot compares final grades across four study-time categories. The median grades are generally higher in the higher study-time categories, although the distributions overlap. This suggests an association between study time and final grades, rather than proving that study time alone causes better performance.

### 7.3 Confusion Matrix

The confusion matrix contains the following results:

| Actual / Predicted | Fail | Pass |
|---|---:|---:|
| Fail | 24 | 2 |
| Pass | 8 | 45 |

The diagonal values represent correct predictions, while the off-diagonal values represent incorrect predictions.

## 8. Limitations

- The dataset represents a specific educational context and may not generalize to all schools or student populations.
- The pass threshold of 10 out of 20 is a project-defined rule.
- The model identifies statistical patterns and cannot guarantee an individual student's future performance.
- The results are based on one train-test split; additional validation would help assess generalization.

## 9. Technologies Used

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## 10. Project Structure

```text
student-performance-prediction/
├── data/
│   └── student-mat.csv
├── outputs/
│   ├── pass_fail_distribution.png
│   ├── studytime_vs_grade.png
│   ├── confusion_matrix.png
│   └── results.txt
├── predictivemodel.py
├── README.md
└── requirements.txt
```

## 11. How to Run the Project

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run the project from the repository's root directory:

```bash
python predictivemodel.py
```

The program prints the dataset information and evaluation metrics and saves the generated graphs and results in the `outputs` directory.

## 12. Conclusion

This project demonstrates the application of logistic regression to a student performance classification problem. The model achieved 87.34% accuracy and a ROC-AUC score of 0.9485 on the test set. Data preprocessing, exploratory visualization, and model evaluation were used to develop and assess the predictive model.

The project illustrates how predictive analytics can support the exploration of academic performance patterns while acknowledging the limitations of the dataset and model.

## 13. Reference

Cortez, P., and Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* Proceedings of the 5th Annual Future Business Technology Conference.

Dataset: UCI Machine Learning Repository, Student Performance. https://archive.ics.uci.edu/dataset/320/student+performance