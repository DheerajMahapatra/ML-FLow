# # The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# # P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# # Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# import os
# import warnings
# import sys

# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import ElasticNet
# from urllib.parse import urlparse
# import mlflow
# from mlflow.models import infer_signature
# import mlflow.sklearn

# import logging

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     np.random.seed(40)

#     # Read the wine-quality csv file from the URL
#     csv_url = (
#         "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
#     )
#     try:
#         data = pd.read_csv(csv_url, sep=";")
#     except Exception as e:
#         logger.exception(
#             "Unable to download training & test CSV, check your internet connection. Error: %s", e
#         )

#     # Split the data into training and test sets. (0.75, 0.25) split.
#     train, test = train_test_split(data)

#     # The predicted column is "quality" which is a scalar from [3, 9]
#     train_x = train.drop(["quality"], axis=1)
#     test_x = test.drop(["quality"], axis=1)
#     train_y = train[["quality"]]
#     test_y = test[["quality"]]

#     alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
#     l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

#     ## For Local tracking (no auth needed)
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")
#     mlflow.set_experiment("ElasticNet Wine Quality")

#     with mlflow.start_run():
#         lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
#         lr.fit(train_x, train_y)

#         predicted_qualities = lr.predict(test_x)

#         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

#         print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
#         print("  RMSE: %s" % rmse)
#         print("  MAE: %s" % mae)
#         print("  R2: %s" % r2)

#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)

#         #predictions = lr.predict(train_x)
#         #signature = infer_signature(train_x, predictions)

#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#         # Model registry does not work with file store
#         if tracking_url_type_store != "file":
#             # Register the model
#             # There are other ways to use the Model Registry, which depends on the use case,
#             # please refer to the doc for more information:
#             # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#             mlflow.sklearn.log_model(
#                 lr, name="model", registered_model_name="ElasticnetWineModel"
#             )
#         else:
#             mlflow.sklearn.log_model(lr, name="model")








import os
import warnings
import sys
import logging

# Windows (cp1252) consoles crash on the unicode characters used by the
# dagshub/rich authorization prompt, so force UTF-8 output first.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
import dagshub


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# =========================================================
# EVALUATION METRICS
# =========================================================

def eval_metrics(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    np.random.seed(40)


    # =====================================================
    # 1. INITIALIZE DAGSHUB
    # =====================================================

    dagshub.init(
        repo_owner="DheerajMahapatra",
        repo_name="ML-FLow",
        mlflow=True
    )


    # =====================================================
    # 2. READ WINE QUALITY DATASET
    # =====================================================

    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/"
        "tests/datasets/winequality-red.csv"
    )

    try:

        data = pd.read_csv(
            csv_url,
            sep=";"
        )

        print("\nDataset loaded successfully!")
        print("Dataset shape:", data.shape)

    except Exception as e:

        logger.exception(
            "Unable to download training & test CSV. "
            "Check your internet connection. Error: %s",
            e
        )

        sys.exit(1)


    # =====================================================
    # 3. TRAIN TEST SPLIT
    # =====================================================

    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=42
    )


    # =====================================================
    # 4. FEATURES AND TARGET
    # =====================================================

    # Target column = quality

    train_x = train.drop(
        ["quality"],
        axis=1
    )

    test_x = test.drop(
        ["quality"],
        axis=1
    )

    train_y = train[["quality"]]

    test_y = test[["quality"]]


    # =====================================================
    # 5. GET MODEL PARAMETERS
    # =====================================================

    alpha = (
        float(sys.argv[1])
        if len(sys.argv) > 1
        else 0.5
    )

    l1_ratio = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 0.5
    )


    # =====================================================
    # 6. SET MLflow EXPERIMENT
    # =====================================================

    mlflow.set_experiment(
        "ElasticNet Wine Quality"
    )


    # =====================================================
    # 7. START MLflow RUN
    # =====================================================

    with mlflow.start_run():

        print("\n----------------------------------------")
        print("MLflow Run Started")
        print("----------------------------------------")


        # =================================================
        # 8. CREATE ELASTIC NET MODEL
        # =================================================

        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )


        # =================================================
        # 9. TRAIN MODEL
        # =================================================

        lr.fit(
            train_x,
            train_y
        )


        # =================================================
        # 10. MAKE PREDICTIONS
        # =================================================

        predicted_qualities = lr.predict(
            test_x
        )


        # =================================================
        # 11. CALCULATE METRICS
        # =================================================

        rmse, mae, r2 = eval_metrics(
            test_y,
            predicted_qualities
        )


        # =================================================
        # 12. PRINT RESULTS
        # =================================================

        print(
            "\nElasticNet model "
            "(alpha={:f}, l1_ratio={:f}):"
            .format(alpha, l1_ratio)
        )

        print(
            "RMSE:",
            rmse
        )

        print(
            "MAE:",
            mae
        )

        print(
            "R2:",
            r2
        )


        # =================================================
        # 13. LOG PARAMETERS TO MLflow
        # =================================================

        mlflow.log_param(
            "alpha",
            alpha
        )

        mlflow.log_param(
            "l1_ratio",
            l1_ratio
        )


        # =================================================
        # 14. LOG METRICS TO MLflow
        # =================================================

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "r2",
            r2
        )


        # =================================================
        # 15. LOG MODEL TO MLflow / DAGSHUB
        # =================================================

        mlflow.sklearn.log_model(
            lr,
            name="model",
            registered_model_name="ElasticnetWineModel"
        )


        # =================================================
        # 16. DISPLAY RUN INFORMATION
        # =================================================

        active_run = mlflow.active_run()

        print("\n----------------------------------------")
        print("MLflow Run Completed")
        print("----------------------------------------")

        print(
            "Run ID:",
            active_run.info.run_id
        )

        print(
            "Experiment ID:",
            active_run.info.experiment_id
        )

        print(
            "Tracking URI:",
            mlflow.get_tracking_uri()
        )

        print("\nExperiment:")
        print("ElasticNet Wine Quality")

        print("\nDagsHub Repository:")
        print("DheerajMahapatra/ML-FLow")

        print("\nYour experiment has been logged to DagsHub.")
