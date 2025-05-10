
# Machine Condition Prediction Using Random Forest

**Name:** KARTHIK.S
**Year:** 2nd Year
**Department:** Mechanical Engineering
**Course:** Data Analysis in Mechanical Engineering
**College:** ARM College of Engineering & Technology

---

## Overview

This project is about predicting the condition of industrial machines using machine learning. I’ve used a trained **Random Forest Classifier** model, which can identify whether a machine is running normally or showing signs of a problem. The prediction is based on important parameters like temperature, vibration, oil quality, RPM, and several other mechanical inputs.

The purpose of this project is to understand how data science methods can be applied in mechanical systems, especially in the area of preventive maintenance and condition monitoring.

---

## Setting Up the Environment

Before running the project, make sure all the required Python libraries are installed. You can install them easily using:

```bash
pip install -r requirements.txt
```

This will install everything needed for data handling, machine learning, and prediction.

---

## Important Files Used

These files are essential for making predictions:

* `random_forest_model.pkl` – This is the saved Random Forest model trained on machine data.
* `scaler.pkl` – A StandardScaler used to normalize the input values.
* `selected_features.pkl` – A list of feature names to ensure inputs match the model expectations.

Make sure all of these files are in your working directory when running predictions.

---

## How the Prediction Process Works

The prediction steps are as follows:

1. **Loading the Necessary Files**

   * The model and scaler are loaded using `joblib.load()`.
   * The list of selected features helps maintain the correct order of inputs.

2. **Preparing the Input**

   * A new data row is created using a Pandas DataFrame.
   * All required feature names must be present and correctly spelled.

3. **Scaling the Input**

   * The scaler transforms the input data to the same format used during training.

4. **Making the Prediction**

   * `.predict()` gives the final predicted class.
   * `.predict_proba()` shows how confident the model is in its prediction.

---

## Example Prediction Script

Here is a sample script (predict.py) you can use to make predictions:

```python
import joblib
import pandas as pd

# Load the model and necessary files
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input (replace with actual values)
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Reorder columns
new_data = new_data[selected_features]

# Scale and predict
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

## Important Points to Remember

* The input features must **match exactly** with those used during training.
* The order of features matters. Do not rearrange or omit columns.
* Make sure the input values are realistic and within expected ranges.

---

## Optional: Updating the Model

If needed, the model can be retrained using the same preprocessing steps:

* Keep the same feature selection and order.
* Reapply the same scaling method.
* Save the updated model and tools using `joblib`.

---

## Applications of This Project

* Predicting if a machine is in *normal* or *faulty* condition.
* Useful in smart factories, maintenance scheduling, and industrial IoT applications.
* Can support real-time condition monitoring in mechanical systems.
