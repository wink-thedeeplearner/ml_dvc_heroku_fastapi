"""
Finalize the model for production by checking its performance on slices
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import src.model_functions
import src.features
import logging


def check_slice_performance():
    """
    Checking model performance on slices
    """
    df = pd.read_csv("data/cleaned/cleaned_census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    slice_values = []

    for cat in src.features.get_cat_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = src.model_functions.feature_engineering(
                df_temp,
                categorical_features=src.features.get_cat_features(),
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = src.model_functions.compute_model_metrics(y_test,
                                                                     y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('data/model/slice_performance_output.txt', 'w') as output:
        for slice_value in slice_values:
            output.write(slice_value + '\n')
