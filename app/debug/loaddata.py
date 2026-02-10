# app/debug/loaddata.py

import os

# from app.loader.datedata import DateLoader
from app.loader.intervaldata import IntervalLoader
from app.utils.filepath import get_data_path
from app.core.logger import setup_logger
logger = setup_logger(__name__)

PATH = os.path.join(get_data_path("fake_data.parquet"))
LABEL = ["y"]
FEATURES = ["f1", "f2", "f3"]
DFFILTER = "date > '2020-01-09'"

TestData = IntervalLoader(
    file=PATH,
    label=LABEL,
    features=FEATURES,
    dffilter=DFFILTER
)

logger.info(f"Label: {TestData.label}")
logger.info(f"Features: {TestData.features}")

logger.info(f"Days: {TestData.keys}")
# logger.info(f"Data: {TestData.data}")

logger.info(f"Length of TestData: {len(TestData)}")

for key, X, y, mask in TestData:
    logger.info(f"Date: {key}, Features: {X[: 5]}, Label: {y[: 5]}, Mask: {mask[: 5]}.")
    break

from datetime import datetime
logger.info(f"Date '2020-01-10' Data: {TestData.get_batch((datetime(2020, 1, 10), 29))}")


# end of app/debug/loaddata.py