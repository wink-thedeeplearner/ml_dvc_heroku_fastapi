"""
Functions for model preparation, building, and testing
"""
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import src.features as features


def feature_engineering(
    X, categorical_features=[], label=None, training=True, encoder=None,
    lb=None
):
    """ Feature Engineering
        - one hot encoding for the categorical features
        - label binarizer for the labels

    TODO: Add scaling for the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label.
        cat_features in src.features.py: list[str]
            List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`.
        If None, return  an empty array for y (default=None)
    training : bool
        Indicate if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Train sklearn OneHotEncoder.
        Use only if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Train sklearn LabelBinarizer.
        Use only if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        If labeled=True, process labels.
        Otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        If training is True, Train OneHotEncode.
        Otherwise returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        If training is True, Train LabelBinarizer
        Otherwise returns the binarizer passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/cleaned/cleaned_census.csv")
    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = feature_engineering(
        train, categorical_features=features.get_cat_features(),
        label="salary", training=True
    )
    trained_model = cv_model(X_train, y_train)

    dump(trained_model, "data/model/model.joblib")
    dump(encoder, "data/model/encoder.joblib")
    dump(lb, "data/model/lb.joblib")


def cv_model(X_train, y_train):
    """
    Train a machine learning model

    Inputs
    ------
    X_train : np.array
        Training data for features.
    y_train : np.array
        Training data fo Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    return model


def compute_model_metrics(y, preds):
    """
    Validate the trained machine learning model using the following metrics:
    - precision, recall, F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds
