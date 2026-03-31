import json
import sys

import pandas as pd
from pandas import DataFrame

from evidently import Report
from evidently.presets import DataDriftPreset

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_categorical_columns:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)

        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """Detect dataset drift using Evidently 0.7.21"""
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)

            report_result = {}
            n_features = 0
            n_drifted_features = 0
            drift_status = False

            for item in report.items():
            # item is the metric object itself in 0.7.21
                item_dict = item.dict() if hasattr(item, 'dict') else vars(item)
                report_result[type(item).__name__] = item_dict

            # Try getting result attribute if it exists
                result = getattr(item, 'result', item)

                if hasattr(result, 'dataset_drift'):
                    drift_status = result.dataset_drift
                if hasattr(result, 'number_of_columns'):
                    n_features = result.number_of_columns
                if hasattr(result, 'number_of_drifted_columns'):
                    n_drifted_features = result.number_of_drifted_columns

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report_result
            )

            logging.info(f"{n_drifted_features}/{n_features} features drifted.")
            logging.info(f"Dataset drift status: {drift_status}")

            return drift_status

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df, test_df = (
                DataValidation.read_data(self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(self.data_ingestion_artifact.test_file_path)
            )

            status = self.validate_number_of_columns(train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.validate_number_of_columns(test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            status = self.is_column_exist(train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.is_column_exist(test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e