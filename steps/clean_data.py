import logging
from typing import Tuple
import pandas as pd
from typing_extensions import Annotated
from zenml import step

# Import your existing DataCleaning classes
from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning step: preprocess and split data into train/test with NaN handling."""
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        # Fill NaNs with median (to avoid pipeline failure)
        x_train = x_train.fillna(x_train.median())
        x_test = x_test.fillna(x_train.median())
        y_train = y_train.fillna(y_train.median())
        y_test = y_test.fillna(y_train.median())

        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
