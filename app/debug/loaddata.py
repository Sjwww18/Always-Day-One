# app/debug/loaddata.py

import os

from app.core.logger import setup_logger
from app.loader.loaddata import LoadData
from app.utils.filepath import get_data_path

logger = setup_logger(__name__)

PATH = os.path.join(get_data_path(), "fake_data.parquet")
LABEL = ["y"]
FEATURES = ["f1", "f2", "f3"]
DFFILTER = "date > '2020-01-02'"
DEVICE = "cpu"

TestData = LoadData(path=PATH, label=LABEL, features=FEATURES, dffilter=DFFILTER, device=DEVICE)

logger.info(f"Label: {TestData.label}")
logger.info(f"Features: {TestData.features}")
logger.info(f"Device: {TestData.device}")

logger.info(f"Days: {TestData.days}")
logger.info(f"Data: {TestData.data}")

logger.info(f"Length of TestData: {len(TestData)}")

for d, X, y in TestData:
    logger.info(f"Date: {d}, Features: {X}, Label: {y}")

logger.info(f"Date '2020-01-03' Data: {TestData.get_date('2020-01-03')}")


# end of app/debug/loaddata.py