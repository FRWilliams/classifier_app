# Income Classification API (Flask)

This is a lightweight Flask API that serves a trained income classification model based on the UCI Adult Income dataset.

## Features
- Accepts JSON input
- Returns a predicted income class (>50K or <=50K)
- Trained using Random Forest with Grid Search and Stratified K-Fold

## How to Run

1. Install dependencies:

## API Endpoint

### POST /predict

**Description:**
Returns a predicted income category based on user demographic features.

**Request Headers:**
- `Content-Type: application/json`

**Request Body Example:**
```json
{
  "age": 27,
  "education_num": 13,
  "hours_per_week": 52,
  "workclass_Private": 1,
  "education_Bachelors": 1,
  "marital_status_Married_civ_spouse": 1,
  "occupation_Exec_managerial": 1,
  "relationship_Husband": 1,
  "race_White": 1,
  "sex_Male": 1,
  "native_country_United_States": 1
}

**Response Body Example:**
'''json
{
  "prediction": ">50K"
}
