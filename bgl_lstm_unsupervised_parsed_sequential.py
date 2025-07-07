import numpy as np

# Add this at the very beginning of your script, before any logai imports
if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.bool = bool

# Monkey patch the problematic function
original_equal = np.equal


def patched_equal(x1, x2, out=None, **kwargs):
    # Remove dtype argument if it's np.int or similar deprecated types
    if 'dtype' in kwargs:
        dtype = kwargs['dtype']
        if dtype == np.int or (hasattr(dtype, '__name__') and dtype.__name__ == 'int'):
            kwargs.pop('dtype')
    return original_equal(x1, x2, out=out, **kwargs)


np.equal = patched_equal
# Now import logai safely
# ... rest of your imports and code
import os
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import OpenSetADWorkflowConfig, validate_config_dict
from logai.utils.file_utils import read_file
from logai.utils.dataset_utils import split_train_dev_test_for_anomaly_detection
import logging
from logai.dataloader.data_loader import FileDataLoader
from logai.preprocess.bgl_preprocessor import BGLPreprocessor
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.openset_partitioner import OpenSetPartitioner
from logai.analysis.nn_anomaly_detector import NNAnomalyDetector
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.utils import constants


config_path = "configs/bgl_lstm_unsupervised_parsed_sequential_config.yaml"
config_parsed = read_file(config_path)
config_dict = config_parsed["workflow_config"]
validate_config_dict(config_dict)
config = OpenSetADWorkflowConfig.from_dict(config_dict)


dataloader = FileDataLoader(config.data_loader_config)
logrecord = dataloader.load_data()
print (logrecord.body[constants.LOGLINE_NAME])


preprocessor = BGLPreprocessor(config.preprocessor_config)
preprocessed_filepath = os.path.join(config.output_dir, 'BGL_11k_processed.csv')
logrecord = preprocessor.clean_log(logrecord)
logrecord.save_to_csv(preprocessed_filepath)
print (logrecord.body[constants.LOGLINE_NAME])


parser = LogParser(config.log_parser_config)
parsed_result = parser.parse(logrecord.body[constants.LOGLINE_NAME])
logrecord.body[constants.LOGLINE_NAME] = parsed_result[constants.PARSED_LOGLINE_NAME]
parsed_filepath = os.path.join(config.output_dir, 'BGL_11k_parsed.csv')
logrecord.save_to_csv(parsed_filepath)
print (logrecord.body[constants.LOGLINE_NAME])


partitioner = OpenSetPartitioner(config.open_set_partitioner_config)
partitioned_filepath = os.path.join(config.output_dir, 'BGL_11k_parsed_sliding10.csv')
logrecord = partitioner.partition(logrecord)
logrecord.save_to_csv(partitioned_filepath)
print (logrecord.body[constants.LOGLINE_NAME])

train_filepath = os.path.join(config.output_dir, 'BGL_11k_parsed_sliding10_unsupervised_train.csv')
dev_filepath = os.path.join(config.output_dir, 'BGL_11k_parsed_sliding10_unsupervised_dev.csv')
test_filepath = os.path.join(config.output_dir, 'BGL_11k_parsed_sliding10_unsupervised_test.csv')

(train_data, dev_data, test_data) = split_train_dev_test_for_anomaly_detection(
                logrecord,training_type=config.training_type,
                test_data_frac_neg_class=config.test_data_frac_neg,
                test_data_frac_pos_class=config.test_data_frac_pos,
                shuffle=config.train_test_shuffle
            )

train_data.save_to_csv(train_filepath)
dev_data.save_to_csv(dev_filepath)
test_data.save_to_csv(test_filepath)
print ('Train/Dev/Test Anomalous', len(train_data.labels[train_data.labels[constants.LABELS]==1]),
                                   len(dev_data.labels[dev_data.labels[constants.LABELS]==1]),
                                   len(test_data.labels[test_data.labels[constants.LABELS]==1]))
print ('Train/Dev/Test Normal', len(train_data.labels[train_data.labels[constants.LABELS]==0]),
                                   len(dev_data.labels[dev_data.labels[constants.LABELS]==0]),
                                   len(test_data.labels[test_data.labels[constants.LABELS]==0]))

vectorizer = LogVectorizer(config.log_vectorizer_config)
vectorizer.fit(train_data)
train_features = vectorizer.transform(train_data)
dev_features = vectorizer.transform(dev_data)
test_features = vectorizer.transform(test_data)


anomaly_detector = NNAnomalyDetector(config=config.nn_anomaly_detection_config)
anomaly_detector.fit(train_features, dev_features)

predict_results = anomaly_detector.predict(test_features)
print (predict_results)

# Check the distribution of your labels
print("Ground truth distribution:")
print(predict_results['true'].value_counts())

print("\nPrediction distribution:")
print(predict_results['pred'].value_counts())

# Look at some normal vs anomaly examples
print("\nFirst 20 predictions:", predict_results['pred'][:20].tolist())
print("First 20 ground truth:", predict_results['true'][:20].tolist())