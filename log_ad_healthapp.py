import os
import sys
import joblib
from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig

#File Configuration
filepath = os.path.join(".", "datasets", "HealthApp.log") # Point to the target HealthApp.log dataset

dataset_name = "HealthApp"
data_loader = OpenSetDataLoader(
    OpenSetDataLoaderConfig(
        dataset_name=dataset_name,
        filepath=filepath)
)

logrecord = data_loader.load_data()

logrecord.to_dataframe().head(5)
print("step 1 complete: log has been loaded and converted to dataframe")

from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.utils import constants

loglines = logrecord.body[constants.LOGLINE_NAME]
attributes = logrecord.attributes

preprocessor_config = PreprocessorConfig(
    custom_replace_list=[
        [r"\d+\.\d+\.\d+\.\d+", "<IP>"],   # retrieve all IP addresses and replace with <IP> tag in the original string.
    ]
)

preprocessor = Preprocessor(preprocessor_config)

clean_logs, custom_patterns = preprocessor.clean_log(
    loglines
)
print("step 2 complete: log has been preprocessed")

from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams

# parsing
parsing_algo_params = DrainParams(
    sim_th=0.5, depth=5
)

log_parser_config = LogParserConfig(
    parsing_algorithm="drain",
    parsing_algo_params=parsing_algo_params
)

parser = LogParser(log_parser_config)
parsed_result = parser.parse(clean_logs)

parsed_loglines = parsed_result['parsed_logline']
print("step 3 complete: log has been parsed with drain algorithm")
print(f"parsed loglines: {parsed_loglines}")

from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor

config = FeatureExtractorConfig(
    group_by_time="15min",
    group_by_category=['parsed_logline', 'Action', 'ID'],
)

feature_extractor = FeatureExtractor(config)

timestamps = logrecord.timestamp['timestamp']
parsed_loglines = parsed_result['parsed_logline']
counter_vector = feature_extractor.convert_to_counter_vector(
    log_pattern=parsed_loglines,
    attributes=attributes,
    timestamps=timestamps
)

print("step 4 complete: parsed loglines have been converted to counter vectors")
print(f"counter_vectors: {counter_vector.head(5)}")

from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig
from sklearn.model_selection import train_test_split
import pandas as pd

counter_vector["attribute"] = counter_vector.drop(
                [
                    constants.LOG_COUNTS,
                    constants.LOG_TIMESTAMPS,
                    constants.EVENT_INDEX
                ],
                axis=1
            ).apply(
                lambda x: "-".join(x.astype(str)), axis=1
            )

attr_list = counter_vector["attribute"].unique()
print(f"attr_list:\n {attr_list}")

anomaly_detection_config = AnomalyDetectionConfig(
    algo_name='dbl'
)

res_list = []  # Changed: Use a list to collect results
for item, attr in enumerate(attr_list):
    # temporary dataframe with only the wanted attribute
    temp_df = counter_vector[counter_vector["attribute"] == attr]
    if temp_df.shape[0] >= constants.MIN_TS_LENGTH:
        train, test = train_test_split(
            temp_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
            shuffle=False,
            train_size=0.3
        )
        anomaly_detector = AnomalyDetector(anomaly_detection_config)
        anomaly_detector.fit(train)
        #  print("Available methods:", [method for method in dir(anomaly_detector) if not method.startswith('_')])

        filepath = os.path.join(".", "models", f"{item}")
        joblib.dump(anomaly_detector, filepath)
        print(f"Model saved to {filepath}.")

        anom_score = anomaly_detector.predict(test)
        res_list.append(anom_score)
        print(f"anom_score = {anom_score}")

# Changed: Use pd.concat() to combine all results
res = pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()

# Get anomalous datapoints
anomalies = counter_vector.iloc[res[res>0].index]
print(f"anomalies\n: {anomalies.head(5)}")