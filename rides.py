import os
import json
from pathlib import Path
from dataclasses import dataclass
from itertools import islice
from time import time
from functools import wraps
from typing import Optional

from tqdm import tqdm
import pandas as pd

from distance import safe_distance


class Folder:
    """Получение JSON файлов из каталога."""

    def __init__(self, directory: str):
        """Use *directory* as JSON file source."""
        self.directory = Path(directory)
        assert self.directory.is_dir()

    def filenames(self):
        return list(self.directory.glob("*.json"))

    def count(self):
        return len(self.filenames())

    def yield_jsons(self):
        for filename in self.filenames():
            yield self.load_json(filename)

    @staticmethod
    def load_json(filepath: str):
        """"Load a JSON file content"""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)


class Ride:
    def __init__(self, dict_):
        self.summary = dict_["info"]
        self.points = dict_["data"]

    @property
    def passengers(self):
        return self.summary["car_passengers"]

    @property
    def vehicle_type(self):
        return self.summary["category"]


@dataclass
class Vehicles:
    directory: str

    def raw(self):
        rides = map(Ride, Folder(self.directory).yield_jsons())
        return pd.DataFrame([r.summary for r in rides])

    def dataframe(self):
        cols = ["category", "car_passengers", "cat_carry_weight"]
        cars_df = self.raw()
        cars_df = cars_df.groupby("car_id").first()[cols].sort_values(cols)
        cars_df["type"] = vehicle_type(cars_df)
        cars_df["qty"] = 1
        return cars_df

    def to_csv(self, csv_path):
        if not os.path.exists(csv_path):
            self.dataframe().to_csv(csv_path)
        else:
            LocalFile(csv_path).warn_file_exists()

    @classmethod
    def read(cls, csv_path):
        dtype = dict(car_id=str, category=str, car_passengers=int, cat_carry_weight=int)
        return pd.read_csv(csv_path, dtype=dtype)


def vehicle_type(cars: pd.DataFrame):
    """Разобрать машины по типам:
        
        bus        
        passenger
        freight
        special
        
    """

    def has(string):
        return lambda s: string in s

    cars["type"] = "other"
    # bus
    cars.loc[cars.car_passengers >= 8, "type"] = "bus"
    # passenger
    a = (cars.category == "Специальный\\Автобус ") & (cars.type == "other")
    b = cars.category.apply(has("Легковой")) & (cars.type == "other")
    cars.loc[(a | b), "type"] = "passenger"
    # freight
    ix = cars.category.apply(has("Грузовой"))
    cars.loc[ix, "type"] = "freight"
    # special
    a = cars.category.apply(has("Специальный")) & (cars.type == "other")
    b = cars.category == "Строительный\\Автокран"
    cars.loc[(a | b), "type"] = "special"
    assert len(cars[cars.type == "other"].category.unique()) == 0
    return cars.type


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f"{f.__name__}() took {te-ts:2.2f} sec")
        return result

    return wrap


def add_distance(rows):
    prev_row = {"ride": ""}
    for row in rows:
        if row["ride"] == prev_row["ride"]:
            row["dist"] = safe_distance(
                row["lat"], row["lon"], prev_row["lat"], prev_row["lon"]
            )
        else:
            row["dist"] = 0
        prev_row = row
        yield row


class Points:
    """Получение точек треков из каталога."""

    columns = ["time", "lon", "lat", "car", "ride"]
    dtypes = dict(time=int, lon=float, lat=float, car=str, ride=str, dist=float)

    def __init__(self, directory: str):
        self.directory = directory

    def iterate(self):
        for dict_ in Folder(self.directory).yield_jsons():
            for p in dict_["data"]:
                car = dict_["info"]["car_id"]
                ride = dict_["info"]["id"]
                yield (p[0], p[1], p[2], car, ride)

    @classmethod
    def to_dict(cls, values):
        return dict((k, v) for k, v in zip(cls.columns, values))

    def islice(self, n: int, a: int):
        return islice(self.iterate(), a, n)

    def _get_iterator(self, n, skip):
        if n is None:
            return self.iterate()
        else:
            return self.islice(n + skip, skip)

    @timing
    def raw_dataframe(self, n=None, skip=0):
        gen = self._get_iterator(n, skip)
        gen = tqdm(gen, unit=" rows")
        return pd.DataFrame(data=gen, columns=self.columns)

    @timing
    def dataframe_with_distances(self, n=None, skip=0):
        gen = self._get_iterator(n, skip)
        gen = tqdm(gen, unit=" rows")
        gen = map(self.to_dict, gen)
        gen = add_distance(gen)
        return pd.DataFrame(gen)

    def to_csv(self, csv_path) -> None:
        if not os.path.exists(csv_path):
            df = self.dataframe_with_distances()
            df.to_csv(csv_path, index=False)
        else:
            LocalFile(csv_path).warn_file_exists()

    @classmethod
    @timing
    def read(cls, csv_path: str, nrows: Optional[int] = None,) -> pd.DataFrame:
        return pd.read_csv(csv_path, nrows=nrows, header=0, dtype=Points.dtypes)


def to_date(x: int):
    return pd.Timestamp(x, unit="s")


@timing
def set_time(df):
    df["time"] = df.time.apply(to_date)


@timing
def set_date(df):
    df["date"] = df["time"].apply(lambda x: x.date())


@timing
def set_time_delta(df):
    df["time_delta"] = df.time.diff()
    # Первая точка любой поездки считается остановкой.
    # Время пребывания в этой остановке равно 0.
    if "prev_ride" not in df.columns:
        df["prev_ride"] = df.ride.shift(+1)
    ix = df.ride != df.prev_ride
    df.loc[ix, "time_delta"] = 0
    assert df.time_delta.min() == 0
    del df["prev_ride"]
    return df


@timing
def merge_with_car_types(df, vehicles_df):
    return df.merge(
        vehicles_df[["car_id", "type"]], left_on="car", right_on="car_id"
    ).drop(columns="car_id")


@timing
def to_canonic(raw_df, vehicles_df):
    # add new data
    df = set_time_delta(raw_df)
    df = merge_with_car_types(df, vehicles_df)
    # convert and decorate
    set_time(df)
    set_date(df)
    return df


@dataclass
class LocalFile:
    path: str

    def warn_file_exists(self):
        print(f"File {self.path} already exists.")


@timing
def get_dataframe_from_jsons(source_folder: str, nrows: Optional[int] = None):
    """Получить типовой (канонический) набор данных из файлов JSON.
    Это более медленный способ, он занимает 5-6 минут.
    """
    print("Reading from JSON files...")
    p = Points(source_folder=source_folder)
    raw_df = p.dataframe_with_distances(nrows=nrows)
    print("Finished reading JSON files, creating dataframe...")
    return to_canonic(raw_df)


@dataclass
class Dataset:
    directory: str
    csv_vehicles: str
    csv_tracks: str

    @property
    def folder(self):
        return Folder(self.directory)

    def prepare_vehicle_info(self, force=False):
        if force:
            os.unlink(self.csv_vehicles)
        Vehicles(self.directory).to_csv(self.csv_vehicles)

    def prepare_tracks_info(self, force=False):
        if force:
            os.unlink(self.csv_tracks)
        Points(self.directory).to_csv(self.csv_tracks)

    @timing
    def full_dataframe(self, nrows: Optional[int] = None):
        self.vs = Vehicles(self.directory)
        vehicle_df = Vehicles.read(self.csv_vehicles)
        raw_df = Points.read(self.csv_tracks)
        return to_canonic(raw_df, vehicle_df)
