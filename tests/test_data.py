from ..utils.data_handling import *
import pandas as pd
from ..utils.data_cleaning import *

def test_get_data():
    data_dir = os.path.join("data","cicids")
    df = get_data(data_dir)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Label" in df.columns

def test_data_processing():
    df = get_sample()
    assert None