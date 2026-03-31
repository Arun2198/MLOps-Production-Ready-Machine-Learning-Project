import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Creating preprocessing pipeline")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("onehot", oh_transformer, oh_columns),
                    ("ordinal", ordinal_encoder, or_columns),
                    ("power", transform_pipe, transform_columns),
                    ("scaler", numeric_transformer, num_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            from us_visa.entity.estimator import TargetValueMapping

            preprocessor = self.get_data_transformer_object()

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # ---------------- TRAIN ----------------
            input_train = train_df.drop(columns=[TARGET_COLUMN])
            target_train = train_df[TARGET_COLUMN]

            input_train['company_age'] = CURRENT_YEAR - input_train['yr_of_estab']

            drop_cols = self._schema_config['drop_columns']
            input_train = drop_columns(df=input_train, cols=drop_cols)

            # ✅ Use .map() and cast to int — ensures SMOTE gets integer labels
            target_mapping = TargetValueMapping()._asdict()
            target_train = target_train.astype(str).str.capitalize().map(target_mapping).astype(int)

            # ---------------- TEST ----------------
            input_test = test_df.drop(columns=[TARGET_COLUMN])
            target_test = test_df[TARGET_COLUMN]

            input_test['company_age'] = CURRENT_YEAR - input_test['yr_of_estab']
            input_test = drop_columns(df=input_test, cols=drop_cols)

            # ✅ Same fix for test target
            target_test = target_test.astype(str).str.capitalize().map(target_mapping).astype(int)

            # ---------------- TRANSFORM ----------------
            input_train_arr = preprocessor.fit_transform(input_train)
            input_test_arr = preprocessor.transform(input_test)

            # ---------------- RESAMPLING (TRAIN ONLY) ----------------
            logging.info("Applying SMOTEENN on training data")

            smt = SMOTEENN(sampling_strategy="minority")

            input_train_final, target_train_final = smt.fit_resample(
                input_train_arr, target_train
            )

            # ❌ DO NOT TOUCH TEST DATA
            input_test_final = input_test_arr
            target_test_final = target_test

            # ---------------- FINAL ARRAYS ----------------
            train_arr = np.c_[input_train_final, np.array(target_train_final)]
            test_arr = np.c_[input_test_final, np.array(target_test_final)]

            # ---------------- SAVE ----------------
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info("Data transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise USvisaException(e, sys) from e