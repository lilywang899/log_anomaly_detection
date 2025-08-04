import os
import numpy as np
import pandas as pd

from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig

def dt64_to_float(dt64):
    year = dt64.astype('M8[Y]')
    days = (dt64 - year).astype('timedelta64[D]')
    year_next = year + np.timedelta64(1, 'Y')
    days_of_year = (year_next.astype('M8[D]') - year.astype('M8[D]')).astype('timedelta64[D]')
    dt_float = 1970 + year.astype(float) + days / (days_of_year)
    return pd.Series(dt_float)

print("Step 1: File Configuration")
filepath = os.path.join(".", "datasets", "HealthApp_2000.log") # Point to the target HealthApp.log dataset

dataset_name = "HealthApp"
data_loader = OpenSetDataLoader(
    OpenSetDataLoaderConfig(
        dataset_name=dataset_name,
        filepath=filepath)
)

logrecord = data_loader.load_data()

print("-------------- log in pandas dataframe structure ----------------")
print(logrecord.to_dataframe().head(5))

print("Step 2: Preprocess")
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

print("Step 3: Parsing")
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

from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer

vectorizer_config = VectorizerConfig(
    algo_name = "word2vec"
)

vectorizer = LogVectorizer(
    vectorizer_config
)

# Train vectorizer
vectorizer.fit(parsed_loglines)

# Transform the loglines into features
log_vectors = vectorizer.transform(parsed_loglines)

print("Step 4: Categorical Encoding for log attributes")
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder

encoder_config = CategoricalEncoderConfig(name="label_encoder")

encoder = CategoricalEncoder(encoder_config)

attributes_encoded = encoder.fit_transform(attributes)

print("Step 5: Feature extraction")
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor

timestamps = logrecord.timestamp['timestamp']

timestamps_float = dt64_to_float(timestamps.to_numpy()) #convert timestamps from datetime dtype to float to allow result_type (common DType)


config = FeatureExtractorConfig(
    max_feature_len=100
)

feature_extractor = FeatureExtractor(config)

_, feature_vector = feature_extractor.convert_to_feature_vector(log_vectors, attributes_encoded, timestamps_float)

from sklearn.model_selection import train_test_split

train, test = train_test_split(feature_vector, train_size=0.7, test_size=0.3)

print("Step 6: Anomaly Detection")
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestParams
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector

algo_params = IsolationForestParams(
    n_estimators=10,
    max_features=100
)
config = AnomalyDetectionConfig(
    algo_name='isolation_forest',
    algo_params=algo_params
)

anomaly_detector = AnomalyDetector(config)
anomaly_detector.fit(train)
res = anomaly_detector.predict(test)

# obtain the anomalous datapoints
anomalies = res[res==1]
print(f"anomalies index: {anomalies.index}")
anomalous_lines = loglines.iloc[anomalies.index].to_frame().join(attributes.iloc[anomalies.index])
print(loglines.iloc[anomalies.index].head(5))
print(attributes.iloc[anomalies.index].head(5))
anomalous_lines.to_csv('./semantic_processing_output/anomaly_detection_results.csv')

print("Step 7: Log Clustering with K-Means clustering algorithm")
from logai.algorithms.clustering_algo.kmeans import KMeansParams
from logai.analysis.clustering import ClusteringConfig, Clustering

clustering_config = ClusteringConfig(
    algo_name='kmeans',
    algo_params=KMeansParams(
        n_clusters=7,
        algorithm='elkan', #elkan or lloyd
    )
)

log_clustering = Clustering(clustering_config)

log_clustering.fit(feature_vector)

cluster_id = log_clustering.predict(feature_vector).astype(str).rename('cluster_id')
cluster_results = logrecord.to_dataframe().join(cluster_id)
cluster_results.to_csv('./semantic_processing_output/kmeans_cluster_results.csv')
# Check clustering results.
print(logrecord.to_dataframe().join(cluster_id).head(5))
