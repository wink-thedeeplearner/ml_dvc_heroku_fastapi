"""
Basic cleaning module test
"""
import pandas as pd
import pytest
import basic_clean


@pytest.fixture
def data():
    """
    Clean the data
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = basic_clean.__clean_dataset(df)
    return df


def test_null(data):
    """
    Check for null values
    """
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    """
    Check for question marks
    """
    assert '?' not in data.values


def test_removed_columns(data):
    """
    Check for removed columns
    """
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns
