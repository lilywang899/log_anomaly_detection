import os

from logai.applications.log_anomaly_detection import LogAnomalyDetection
from logai.applications.application_interfaces import WorkFlowConfig
import json

# path to json configuration file
json_config = os.path.join(".", "configs", "log_anomaly_detection_config.json")

# Create log anomaly detection application workflow configuration
with open(json_config, 'r') as f:
    config = json.load(f)

workflow_config = WorkFlowConfig.from_dict(config)

# Create LogAnomalyDetection Application for given workflow_config
app = LogAnomalyDetection(workflow_config)

# Execute App
app.execute()

print(app.anomaly_results.head(5))
