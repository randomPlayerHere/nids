from ..utils.data_handling import *
import pandas as pd

def test_get_data():
    data_dir = os.path.join("data","cicids")
    df = get_data(data_dir)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Label" in df.columns

if __name__ == "__main__":
    print("Working")