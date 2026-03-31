from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USvisaException
import pandas as pd
import sys
from typing import Optional
import numpy as np


class USvisaData:

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise USvisaException(e, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            # ✅ Select DB correctly
            if database_name is None:
                database = self.mongo_client.database
            else:
                database = self.mongo_client.client[database_name]

            collection = database[collection_name]

            # ✅ Fetch data properly
            data = list(collection.find({}, {"_id": 0}))

            # 🔥 DEBUG (CRITICAL)
            print("Database:", DATABASE_NAME)
            print("Collection:", collection_name)
            print("Records fetched:", len(data))

            # ❌ HARD FAIL if empty
            if len(data) == 0:
                raise Exception("No records found in MongoDB collection")

            df = pd.DataFrame(data)

            df.replace({"na": np.nan}, inplace=True)

            print("DataFrame shape:", df.shape)

            return df

        except Exception as e:
            raise USvisaException(e, sys)