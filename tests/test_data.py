from src.data_preprocessing import load_and_clean_data


def test_data_load():
    df = load_and_clean_data("data/heart.csv")
    assert len(df) > 0
    assert "target" in df.columns
