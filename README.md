# AutoJudge: Predicting Programming Problem Difficulty

## Project Overview
AutoJudge is a Machine Learning–based system that predicts the difficulty of programming problems using **only textual information**.  
Given a problem’s description, input format, and output format, the system predicts:

- **Difficulty Class**: Easy / Medium / Hard (Classification)
- **Difficulty Score**: Numerical difficulty value (Regression)

This project simulates how online coding platforms (like Codeforces, CodeChef, LeetCode) estimate problem difficulty, but does so automatically without human intervention.

---

## Dataset Used
The dataset was **provided by the college** in JSONL format.  
Each problem contains the following fields:

- `title`
- `description`
- `input_description`
- `output_description`
- `sample_io`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (Numerical difficulty score)
- `url`

Only **textual fields** were used for prediction, as required by the problem statement.

---

## Approach and Models Used

### 1. Data Preprocessing
- Combined all textual fields into a single column (`full_text`)
- Converted text to lowercase
- Removed unwanted characters
- Handled empty strings explicitly (not treated as NaN by default)

---

### 2. Feature Engineering
The following features were extracted from text:

- **TF-IDF vectors** (unigrams + bigrams)
- Text length
- Number of mathematical symbols (`+ - * / < > =`)
- Keyword frequencies:
  - `dp`, `graph`, `tree`, `greedy`, `math`, `string`

TF-IDF features and numeric features were combined into a single feature matrix.

---

### 3. Classification Model
- **Model**: Logistic Regression
- **Task**: Predict problem difficulty class (Easy / Medium / Hard)

**Evaluation Metric**:
- Accuracy
- Confusion Matrix

**Final Classification Accuracy**:
47.4%
This result was selected after comparing multiple models (Logistic Regression, Linear SVC, class-weighted variants).

---

### 4. Regression Model
- **Model**: Random Forest Regressor
- **Task**: Predict numerical difficulty score

**Evaluation Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

**Final Regression Results**:
MAE: 1.726
RMSE: 2.047

## Web Interface (Streamlit)

A simple **Streamlit web application** was built to demonstrate the model.

### Features:
- Text input fields for:
  - Problem description
  - Input description
  - Output description
- Predict button
- Displays:
  - Predicted difficulty class
  - Predicted difficulty score

The UI uses the **same trained models** used during evaluation.
