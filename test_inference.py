"""
Inference test for Titanic classification models.

This script verifies that the trained pipelines can be loaded
and used for prediction with correct input features.
"""

import joblib
import pandas as pd

MODEL_PATHS = [
    "models/global_best_model.pkl",
    "models/global_best_model_optuna.pkl",
]

# Sample Titanic passenger (MUST MATCH PIPELINE FEATURES)
sample_input = pd.DataFrame([{
    "Pclass": 3,
    "Age": 22,
    "Fare": 7.25,
    "SibSp": 1,
    "Parch": 0,
    "sex": "male",
}])

print("\nTesting Titanic inference...\n")

passed = 0

for path in MODEL_PATHS:
    print("=" * 70)
    print(f"Testing model: {path}")
    print("=" * 70)

    try:
        model = joblib.load(path)
        print("✓ Model loaded")

        print("Pipeline steps:", list(model.named_steps.keys()))

        prediction = model.predict(sample_input)
        print("✓ Prediction:", int(prediction[0]))

        passed += 1

    except Exception as e:
        print("✗ FAILED:", str(e))

print("\n" + "=" * 70)
print(f"SUMMARY: {passed}/{len(MODEL_PATHS)} models passed inference")
print("=" * 70)
