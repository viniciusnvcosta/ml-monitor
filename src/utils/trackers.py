import datetime
import mlflow
import mlflow.sklearn
import cloudpickle as pickle
import numpy as np
import pandas as pd
from sklearn import metrics

# import aws_wrangler as wr
# import boto3


class Tracker:
    """
    Base class for tracking models and metrics.

    This class serves as the base for all model tracking and metric calculation tasks in the MLOps system. It provides a factory method to create instances of specific subclasses based on the tracker type. Subclasses implement the tracking logic and define methods for calculating metrics, retrieving model information, and saving metrics.

    Usage:
    - Use the factory method `factory()` to create an instance of the appropriate tracker subclass based on the tracker type.
    - Call the `track()` method on the created instance to perform the tracking and metric calculation tasks.
    - Implement the abstract methods `calculate_metrics()`, `get_model()`, and `save_metrics()` in the subclasses to define the specific tracking and metric calculation logic.

    Attributes:
        None

    Methods:
        factory(tracker_type):
            Factory method to create an instance of a specific tracker subclass based on the tracker type.

        track():
            Perform the tracking and metric calculation tasks. Subclasses must implement this method to define the specific tracking logic.

        calculate_metrics(y_true, y_pred):
            Abstract method to calculate metrics based on the true and predicted values. Subclasses must implement this method to define the metric calculation logic.

        get_model(model_path):
            Abstract method to retrieve model information, such as the project and tenant info, from the model path. Subclasses must implement this method to define the model retrieval logic.

        save_metrics(metrics):
            Abstract method to save the calculated metrics. Subclasses must implement this method to define the metric saving logic.

    Subclasses:
        - TrackRegression: Subclass for regression tracking and metric calculation.
        - TrackClassification: Subclass for classification tracking and metric calculation.
        - TrackClustering: Subclass for clustering tracking and metric calculation.
        - TrackRecommendation: Subclass for recommendation tracking and metric calculation.
        - TrackTimeseries: Subclass for time series tracking and metric calculation.
    """

    # Factory method
    @classmethod
    def factory(cls, tracker_type):
        """
        Factory method to create an instance of a specific tracker subclass based on the tracker type.
        """
        available_trackers = {
            "regression": TrackRegression,
            "classification": TrackClassification,
            "clustering": TrackClustering,
            "recommendation": TrackRecommendation,
            "timeseries": TrackTimeseries,
        }
        if tracker_type not in available_trackers:
            raise ValueError("Unknown tracker type")

        return available_trackers[tracker_type]()

    # Track method
    def track(self, model_path, project_id, tenant_id, y_true, y_pred):
        """
        Track the model and metrics.

        Args:
            model_path (str): Path to the model file.
            project_id (str): ID of the project.
            tenant_id (str): ID of the tenant.
            y_true: True values of the target variable.
            y_pred: Predicted values of the target variable.
        """

        self.get_model(model_path, project_id, tenant_id)

        track_metrics = self.calculate_metrics(y_true, y_pred)

        self.save_metrics(track_metrics)

    def get_model(self, model_path, project_id, tenant_id):
        """
        Retrieve model information, such as the project and tenant info, from the model path.
        """
        # Get model UUID from path
        model_uuid = model_path.split("/")[-1]
        # Set model UUID as tag
        mlflow.set_tag("model_uuid", model_uuid)
        # Set tenant/project ID as tag
        mlflow.set_tag("tenant_id", tenant_id)
        mlflow.set_tag("project_id", project_id)
        # Generate version number based on current timestamp
        version_number = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Set version number as tag
        mlflow.set_tag("version", version_number)
        # Return flag to indicate success
        return True

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate metrics based on the true and predicted values.
        Implemented in the subclasses.
        """
        raise NotImplementedError("Method not implemented in subclass")

    def save_metrics(self, track_metrics):
        """
        Save the calculated metrics.
        """
        # Log the metrics to the MLOps tracking system (mlflow)
        for metric_name, metric_value in track_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        # Return flag to indicate success
        return True


class TrackRegression(Tracker):
    """
    define relevant metrics for regression
    """

    def calculate_metrics(self, y_true, y_pred):
        reg_metrics = {
            "r2": metrics.r2_score(y_true, y_pred),
            "mape": metrics.mean_absolute_percentage_error(y_true, y_pred),
            "mse": metrics.mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        }
        return reg_metrics


class TrackClassification(Tracker):
    """
    define relevant metrics for classification
    """

    def calculate_metrics(self, y_true, y_pred):
        class_metrics = {
            "precision": metrics.precision_score(y_true, y_pred),
            "recall": metrics.recall_score(y_true, y_pred),
            "f1-score": metrics.f1_score(y_true, y_pred),
            "auc": metrics.roc_auc_score(y_true, y_pred),
        }
        return class_metrics


class TrackClustering(Tracker):
    """
    define relevant metrics for clustering
    """

    def calculate_metrics(self, y_true, y_pred):
        cluster_metrics = {
            "silhouette": metrics.silhouette_score(y_true, y_pred),
            "calinski_harabasz": metrics.calinski_harabasz_score(y_true, y_pred),
            "davies_bouldin": metrics.davies_bouldin_score(y_true, y_pred),
        }
        return cluster_metrics


class TrackRecommendation(Tracker):
    """
    define relevant metrics for recommendation systems
    """

    def calculate_metrics(self, y_true, y_pred):
        rec_metrics = {
            "precision": metrics.precision_score(y_true, y_pred),
            "recall": metrics.recall_score(y_true, y_pred),
            "f1-score": metrics.f1_score(y_true, y_pred),
            "map": metrics.average_precision_score(y_true, y_pred),
            "ndcg": metrics.ndcg_score(y_true, y_pred),
        }
        return rec_metrics


class TrackTimeseries(Tracker):
    """
    define relevant metrics for time series prediction
    """

    def calculate_metrics(self, y_true, y_pred):
        ts_metrics = {
            "mape": metrics.mean_absolute_percentage_error(
                y_true, y_pred
            ),  # mean absolute percentage error
            "smape": metrics.mean_absolute_percentage_error(y_true, y_pred) * 200.0,
            # symmetric mean absolute percentage error
            "rmse": np.sqrt(
                metrics.mean_squared_error(y_true, y_pred)
            ),  # root mean squared error
            "mfe": np.mean(y_pred - y_true),  # mean forecast error
        }
        return ts_metrics
