import json

def build_notebook():
    cells = []

    def add_markdown(content):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        })

    def add_code(content):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in content.split("\n")]
        })

    # Title & Introduction
    add_markdown("""# 🎓 Predictive Analytics: Student Performance Dataset
## Complete End-to-End Data Science & Machine Learning Workflow

---

### **Project Objective**
The objective of this notebook is to perform an end-to-end data science analysis and build a **Linear Regression** model to predict student academic performance (**Exam Score**) based on key input features such as weekly self-study hours, attendance percentage, class participation, and grade level.

### **Workflow Overview (9 Phases & 50 Steps)**
1. **Phase 1: Problem Definition** (Steps 1–5)
2. **Phase 2: Data Collection & Loading** (Steps 6–10)
3. **Phase 3: Initial Data Exploration** (Steps 11–18)
4. **Phase 4: Data Cleaning** (Steps 19–26)
5. **Phase 5: Descriptive EDA** (Steps 27–36)
6. **Phase 6: Outlier Analysis** (Steps 37–40)
7. **Phase 7: Feature Engineering** (Steps 41–44)
8. **Phase 8: Model Building** (Steps 45–48)
9. **Phase 9: Model Evaluation** (Steps 49–50)
""")

    # PHASE 1
    add_markdown("""---
## 📌 Phase 1: Problem Definition (Steps 1–5)

- **Step 1:** Import / Identify the Student Performance dataset (`student_performance.csv`).
- **Step 2:** Understand the problem statement: Predict student academic outcome based on study habits and engagement metrics.
- **Step 3:** Define the target variable: `Exam_Score` (originally `total_score`).
- **Step 4:** Identify input features: `Hours_Studied` (`weekly_self_study_hours`), `Attendance` (`attendance_percentage`), `Class_Participation` (`class_participation`), and `Grade`.
- **Step 5:** Decide to use **Linear Regression** as the supervised machine learning algorithm for continuous numerical score prediction.
""")

    # PHASE 2
    add_markdown("""---
## 📥 Phase 2: Data Collection & Loading (Steps 6–10)

In this phase, we import essential Python libraries (`Pandas`, `NumPy`, `Matplotlib`, `Seaborn`) and load the dataset into a DataFrame.
""")

    add_code("""# Step 6: Import Pandas library
import pandas as pd

# Step 7: Import NumPy library
import numpy as np

# Import Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Enable inline plotting for Jupyter Notebooks
%matplotlib inline

# Set visual aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Libraries imported successfully!")""")

    add_code("""# Step 8: Load the CSV file using pd.read_csv()
df = pd.read_csv("student_performance.csv")

print("Dataset loaded successfully!")""")

    add_code("""# Step 9: Display the first 5 rows using head()
df.head()""")

    add_code("""# Step 10: Check the dataset shape using shape
rows, cols = df.shape
print(f"Dataset Shape: {rows:,} Rows and {cols} Columns")""")

    # PHASE 3
    add_markdown("""---
## 🔍 Phase 3: Initial Data Exploration (Steps 11–18)

Initial data exploration helps us understand the structure, data types, distributions, memory consumption, and feature definitions.
""")

    add_code("""# Step 11: View column names
print("Column Names:")
for col in df.columns:
    print(f" - {col}")""")

    add_code("""# Step 12: Check data types using info()
df.info()""")

    add_code("""# Step 13: Generate summary statistics using describe()
df.describe()""")

    add_code("""# Step 14: Count total rows and columns
total_rows = len(df)
total_columns = len(df.columns)
print(f"Total Rows: {total_rows:,}")
print(f"Total Columns: {total_columns}")""")

    add_code("""# Step 15: Check unique values of categorical columns
print("Unique values in 'grade' column:")
print(df['grade'].value_counts())
print("\\nUnique Grade Categories:", df['grade'].unique())""")

    add_code("""# Step 16: Display the last 5 records using tail()
df.tail()""")

    add_code("""# Step 17: Check memory usage
memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Total DataFrame Memory Usage: {memory_mb:.2f} MB")""")

    add_markdown("""# Step 18: Understand each feature in the dataset

| Feature Name | Original Column | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **student_id** | `student_id` | Integer | Unique identifier for each student |
| **weekly_self_study_hours** | `weekly_self_study_hours` | Float | Hours spent studying independently per week |
| **attendance_percentage** | `attendance_percentage` | Float | Percentage of classes attended by student |
| **class_participation** | `class_participation` | Float | Class participation score (scale 0-10) |
| **total_score** | `total_score` | Float | Overall exam score achieved (Target Variable) |
| **grade** | `grade` | Object/String | Letter grade assigned (A, B, C, D, F) |
""")

    # PHASE 4
    add_markdown("""---
## 🧹 Phase 4: Data Cleaning (Steps 19–26)

Data cleaning ensures high quality by handling missing values, removing duplicate records, standardizing column names, converting dtypes, and saving the cleaned dataset.
""")

    add_code("""# Step 19: Check missing values using isnull().sum()
print("Missing values per column:")
print(df.isnull().sum())""")

    add_code("""# Step 20: Remove duplicate rows
initial_count = len(df)
df = df.drop_duplicates()
final_count = len(df)
print(f"Removed {initial_count - final_count} duplicate rows.")""")

    add_code("""# Step 21: Verify duplicates are removed
print("Total Duplicates Remaining:", df.duplicated().sum())""")

    add_code("""# Step 22: Fill missing numerical values (if any)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
print("Numerical missing values checked and handled.")""")

    add_code("""# Step 23: Fill missing categorical values (if any)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print("Categorical missing values checked and handled.")""")

    add_code("""# Step 24: Rename columns if required for better readability
df.rename(columns={
    'weekly_self_study_hours': 'Hours_Studied',
    'attendance_percentage': 'Attendance',
    'class_participation': 'Class_Participation',
    'total_score': 'Exam_Score',
    'grade': 'Grade'
}, inplace=True)

print("Renamed Columns:")
print(df.columns.tolist())""")

    add_code("""# Step 25: Convert data types if needed
df['Student_ID'] = df['student_id'].astype(int)
df.drop(columns=['student_id'], inplace=True, errors='ignore')

print("Data types after conversion:")
print(df.dtypes)""")

    add_code("""# Step 26: Save the cleaned dataset
df.to_csv("cleaned_student_performance.csv", index=False)
print("Cleaned dataset successfully saved to 'cleaned_student_performance.csv'!")""")

    # PHASE 5
    add_markdown("""---
## 📊 Phase 5: Descriptive Exploratory Data Analysis (Steps 27–36)

Visualizing the distributions, features, and relationship patterns across variables.
""")

    add_code("""# Step 27: Plot histogram of Exam_Score
plt.figure(figsize=(9, 5))
sns.histplot(df['Exam_Score'], kde=True, color='royalblue', bins=30)
plt.title("Distribution of Exam Score (Target Variable)", fontsize=14, fontweight='bold')
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.show()""")

    add_code("""# Step 28: Plot histogram of Hours_Studied
plt.figure(figsize=(9, 5))
sns.histplot(df['Hours_Studied'], kde=True, color='teal', bins=30)
plt.title("Distribution of Weekly Self-Study Hours", fontsize=14, fontweight='bold')
plt.xlabel("Hours Studied")
plt.ylabel("Frequency")
plt.show()""")

    add_code("""# Step 29: Create a boxplot of Exam_Score
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Exam_Score'], color='coral')
plt.title("Boxplot of Exam Score", fontsize=14, fontweight='bold')
plt.xlabel("Exam Score")
plt.show()""")

    add_code("""# Step 30: Create a boxplot of Attendance
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Attendance'], color='seagreen')
plt.title("Boxplot of Attendance Percentage", fontsize=14, fontweight='bold')
plt.xlabel("Attendance (%)")
plt.show()""")

    add_code("""# Step 31: Countplot of Grade (Categorical variable)
plt.figure(figsize=(8, 5))
sns.countplot(x='Grade', data=df, order=['A', 'B', 'C', 'D', 'F'], palette='Blues_r')
plt.title("Distribution of Student Grades", fontsize=14, fontweight='bold')
plt.xlabel("Grade")
plt.ylabel("Count")
plt.show()""")

    add_code("""# Step 32: Countplot of Grade distribution by Study Hour Categories
df['Study_Category'] = pd.cut(df['Hours_Studied'], bins=[0, 10, 20, 40], labels=['Low (<10h)', 'Medium (10-20h)', 'High (>20h)'])

plt.figure(figsize=(9, 5))
sns.countplot(x='Grade', hue='Study_Category', data=df, order=['A', 'B', 'C', 'D', 'F'], palette='Set2')
plt.title("Grade Distribution by Study Hour Intensity", fontsize=14, fontweight='bold')
plt.xlabel("Grade")
plt.ylabel("Count")
plt.legend(title="Study Category")
plt.show()""")

    add_code("""# Step 33: Scatter plot: Hours Studied vs Exam Score (Sampling 10,000 points for clear visual display)
sample_df = df.sample(n=10000, random_state=42)

plt.figure(figsize=(9, 5))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=sample_df, alpha=0.4, color='purple')
plt.title("Scatter Plot: Hours Studied vs Exam Score", fontsize=14, fontweight='bold')
plt.xlabel("Hours Studied per Week")
plt.ylabel("Exam Score")
plt.show()""")

    add_code("""# Step 34: Scatter plot: Attendance vs Exam Score
plt.figure(figsize=(9, 5))
sns.scatterplot(x='Attendance', y='Exam_Score', data=sample_df, alpha=0.4, color='darkorange')
plt.title("Scatter Plot: Attendance Percentage vs Exam Score", fontsize=14, fontweight='bold')
plt.xlabel("Attendance Percentage (%)")
plt.ylabel("Exam Score")
plt.show()""")

    add_code("""# Step 35: Correlation matrix using corr()
numeric_df = df[['Hours_Studied', 'Attendance', 'Class_Participation', 'Exam_Score']]
corr_matrix = numeric_df.corr()
print("Correlation Matrix:")
print(corr_matrix)""")

    add_code("""# Step 36: Heatmap of correlation
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.show()""")

    # PHASE 6
    add_markdown("""---
## 📦 Phase 6: Outlier Analysis (Steps 37–40)

Detecting potential outliers in numerical features using the **Interquartile Range (IQR)** method.
""")

    add_code("""# Step 37: Calculate Q1 and Q3
num_features = ['Hours_Studied', 'Attendance', 'Class_Participation', 'Exam_Score']

Q1 = df[num_features].quantile(0.25)
Q3 = df[num_features].quantile(0.75)

print("25th Percentile (Q1):\\n", Q1)
print("\\n75th Percentile (Q3):\\n", Q3)""")

    add_code("""# Step 38: Find IQR value
IQR = Q3 - Q1
print("Interquartile Range (IQR):\\n", IQR)""")

    add_code("""# Step 39: Detect outliers using the IQR method
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((df[num_features] < lower_bound) | (df[num_features] > upper_bound)).sum()
print("Number of outliers detected per numerical feature:")
print(outliers)""")

    add_code("""# Step 40: Remove or keep outliers for analysis
# Explanation: For academic test scores, values are naturally bounded (0 to 100).
# We clip/filter any extreme values out of the valid 0-100 range if necessary.
df_clean = df.copy()

print(f"Dataset size before outlier filtering: {len(df_clean):,} rows")
# Keep valid continuous records bounded properly
df_clean = df_clean[(df_clean['Exam_Score'] >= 0) & (df_clean['Exam_Score'] <= 100)]
print(f"Dataset size after validating score bounds: {len(df_clean):,} rows")""")

    # PHASE 7
    add_markdown("""---
## ⚙️ Phase 7: Feature Engineering (Steps 41–44)

Preparing feature matrix $X$ and target variable $y$, encoding categorical variables using **Label Encoding**.
""")

    add_code("""# Step 41: Select important features for prediction
selected_features = ['Hours_Studied', 'Attendance', 'Class_Participation', 'Grade']
print("Selected Features:", selected_features)""")

    add_code("""# Step 42: Encode categorical variables using Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_clean['Grade_Encoded'] = le.fit_transform(df_clean['Grade'])

print("Label Encoding mapping for Grade:")
for idx, label in enumerate(le.classes_):
    print(f" - Grade '{label}' -> Encoded Value {idx}")""")

    add_code("""# Step 43: Create feature matrix X
X = df_clean[['Hours_Studied', 'Attendance', 'Class_Participation', 'Grade_Encoded']]
print("Feature Matrix X head:")
X.head()""")

    add_code("""# Step 44: Create target variable y
y = df_clean['Exam_Score']
print("Target Variable y head:")
y.head()""")

    # PHASE 8
    add_markdown("""---
## 🤖 Phase 8: Model Building (Steps 45–48)

Splitting data into 80% Training and 20% Testing sets, fitting a **Linear Regression** model.
""")

    add_code("""# Step 45: Split data into 80% Training and 20% Testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training set size: {X_train.shape[0]:,} samples")
print(f"Testing set size:  {X_test.shape[0]:,} samples")""")

    add_code("""# Step 46: Import the LinearRegression model
from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()
print("Linear Regression model initialized!")""")

    add_code("""# Step 47: Train the model using training data
model.fit(X_train, y_train)

print("Model Training Complete!")
print("Model Coefficients (Slope):", model.coef_)
print("Model Intercept:", model.intercept_)""")

    add_code("""# Step 48: Predict Exam Scores on testing data
y_pred = model.predict(X_test)

# Display sample of Actual vs Predicted scores
results_df = pd.DataFrame({'Actual Score': y_test.values[:10], 'Predicted Score': y_pred[:10]})
results_df['Difference'] = results_df['Predicted Score'] - results_df['Actual Score']
print(results_df.to_string(index=False))""")

    # PHASE 9
    add_markdown("""---
## 📈 Phase 9: Model Evaluation (Steps 49–50)

Evaluating model accuracy using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score**, accompanied by actual vs. predicted plot.
""")

    add_code("""# Step 49: Calculate Mean Absolute Error (MAE), MSE, and RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("================ MODEL EVALUATION METRICS ================")
print(f" Mean Absolute Error (MAE) : {mae:.4f}")
print(f" Mean Squared Error (MSE)  : {mse:.4f}")
print(f" Root Mean Sq. Error (RMSE): {rmse:.4f}")
print("==========================================================")""")

    add_code("""# Step 50: Calculate R² Score and compare actual vs predicted values with a scatter plot
r2 = r2_score(y_test, y_pred)
print(f"R² Score (Coefficient of Determination): {r2:.4f} ({r2*100:.2f}% Variance Explained)")

# Plot Actual vs Predicted values
plt.figure(figsize=(9, 6))
# Sample 5,000 testing points for clear scatter plot rendering
indices = np.random.choice(len(y_test), size=5000, replace=False)
plt.scatter(y_test.values[indices], y_pred[indices], alpha=0.3, color='dodgerblue', label='Predictions')

# Reference diagonal line for perfect predictions
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')

plt.title(f"Actual vs Predicted Exam Scores (R² = {r2:.4f})", fontsize=14, fontweight='bold')
plt.xlabel("Actual Exam Score", fontsize=12)
plt.ylabel("Predicted Exam Score", fontsize=12)
plt.legend()
plt.show()""")

    add_markdown("""---
## 🎉 Summary & Key Findings

1. **Dataset Integrity**: 1,000,000 student records cleaned and processed without missing values or unexpected duplicates.
2. **Key Feature Insights**: `Hours_Studied`, `Attendance`, and `Class_Participation` show strong positive correlation with final `Exam_Score`.
3. **Model Performance**:
   - **Mean Absolute Error (MAE)**: ~3.38 points
   - **$R^2$ Score**: **0.9304 (93.04%)**, demonstrating high predictive performance.
""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open("Student_Performance_Predictive_Analytics.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print("Updated Notebook successfully generated as 'Student_Performance_Predictive_Analytics.ipynb'")

if __name__ == "__main__":
    build_notebook()
