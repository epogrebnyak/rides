from rides import Dataset

# 'sample_jsons' folder has three files


def test_rides():
    d = Dataset("sample_jsons", "temp")
    d.build()
    types = d.vehicle_types()
    assert types == ["bus", "freight"]
    df = d.get_dataframe(10)
    assert df.shape == (10, 9)
    # assert df.shape == (5174, 9)
