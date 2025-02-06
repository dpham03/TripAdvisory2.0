import os
import pandas as pd
import networkx as nx
import requests
import pickle
from constants import HTTP

def conditionally_fetch_from_web(fetch_from_web, filename, url, columns, usecols=None):
    """Fetches data from the web and saves it as a pickle file if it doesn't already exist."""
    if not fetch_from_web:
        return load_pkl(filename)
    return load_pkl_or_build_from_web(filename, url, columns, usecols)

def load_pkl_or_build_from_web(filename, url, columns, usecols=None):
    """Loads the data from pickle if available; otherwise, fetches it from the web."""
    pickled_data = load_pkl(filename)
    if pickled_data is not None:
        return pickled_data
    else:
        usecols=columns if usecols is None else usecols
        response = requests.get(url)
        if response.status_code == HTTP.OK:
            df = pd.read_csv(url, names=columns, usecols=usecols)
            df.to_pickle(filename)
            return df
        else:
            return None

def load_pkl(filename):
    """Loads the data from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            return None

def store_pkl(filename, data):
    """Stores data in a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)