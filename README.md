## ML FLow experiements

import dagshub
dagshub.init(repo_owner='DheerajMahapatra', repo_name='ML-FLow', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)