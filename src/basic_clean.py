"""
ML step: basic cleaning
"""
import pandas as pd


def __clean_dataset(df):
    """
    preprocess the raw data after data exploration
    """
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    return df


def execute_basic_cleaning():
    """
    Execute data cleaning
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = __clean_dataset(df)
    df.to_csv("data/cleaned/cleaned_census.csv", index=False)
