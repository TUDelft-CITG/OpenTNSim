import pathlib

import pandas as pd


def get_expected_df_name(path):
    """get the expected dataframe name, given the filename (with _expected.csv at the end)"""
    name = f"{path.stem}_expected.csv"
    return name


def get_expected_df(path):
    """return a dataframe based on a csv file with the path to the test but then with _expected.csv"""
    name = get_expected_df_name(path)
    # filename but then with _expected.csv at the end
    path = path.with_name(name)
    df = pd.read_csv(path)
    return df
