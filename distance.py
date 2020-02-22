"""Working with geographic distances."""
import pandas as pd
import numpy as np
from geopy.distance import great_circle


def distance_km(a: tuple, b: tuple):
    # great_circle - менее точен, но быстрее чем geopy.distance.distance
    # https://geopy.readthedocs.io/en/stable/#module-geopy.distance
    return great_circle(a, b).km


def safe_distance(lat, lon, prev_lat, prev_lon):
    a = (lat, lon)
    b = (prev_lat, prev_lon)
    if a == b:
        return 0
    try:
        return distance_km(a, b)
    except ValueError:
        return np.nan
