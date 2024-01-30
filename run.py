import os
import uuid
import warnings

import mlflow
import pandas as pd
from utils import Trackers
import yaml

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read config file
    with open("src/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Iterate over projects in the config file
    for project in config:
        project_id = project['project_id']

        # Iterate over tenants in the project
        for tenant in project['tenants']:
            tenant_id = tenant['tenant_id']

            # Iterate over model types for the tenant
            for model_type in tenant['model_types']:
                model_type_name = model_type['type']

                # Iterate over models for the model type
                for model in model_type['models']:
                    model_name = model['model_name']
                    data = model['data']

                    # Read data file
                    data = pd.read_parquet(data)

                    # Get y_true and y_pred
                    y_true = data['y_true']
                    y_pred = data['y_pred']

                    # Get timestamp
                    timestamp = data['last_update'].max()

                    # Create a "commit-like" run uuid based on the model name
                    run_uuid = model_name + "-" + str(uuid.uuid4())

                    # Set mlflow experiment_id as model_name
                    with mlflow.start_run(
                        experiment_id=model_name,
                        run_name=run_uuid,
                        tags={
                            "project_id": project_id,
                            "tenant_id": tenant_id,
                            "paradigm": model_type_name,
                            "last_update": timestamp,
                        },
                    ):
                        # Find the tracker type through the model path
                        # Assuming `model_path` variable is defined somewhere in the code
                        model_path = '<path_to_your_model_file>'
                        model_type = Trackers.find_model_type(model_path)

                        # Create an instance of the appropriate Tracker subclass
                        tracker = Trackers.factory(model_type)

                        # Execute and Log metrics on mlflow
                        tracker.track(model_path, project_id, tenant_id, y_true, y_pred)
