from rides import Dataset

# 'sample_jsons' folder has three files


def test_rides():
    d = Dataset("sample_jsons", "vehicles.csv", "df.csv")
    d.prepare_vehicle_info()
    d.prepare_tracks_info()
    df = d.full_dataframe()
    assert df.shape == (5174, 9)
