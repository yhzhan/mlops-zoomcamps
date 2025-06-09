import mlflow.sklearn
from prefect import flow, task, get_run_logger
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow


@task(log_prints=True)
def download_data(url: str) -> pd.DataFrame:
    """Download data from the given URL and return a DataFrame."""
    # Question 3
    logger = get_run_logger()
    logger.info(f"Downloading data from {url}")
    df = pd.read_parquet(url)
    logger.info(f"Downloaded {len(df)} rows")
    return df

@task(log_prints=True)
def read_dataframe(filename):
    """Read a DataFrame from a Parquet file and preprocess it."""
    # Question 4
    logger = get_run_logger()
    logger.info(f"Reading data from {filename}")
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    logger.info(f"Cleaned data: {len(df)} rows after filtering")
    return df

@task(log_prints=True)
def train_model(df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Training model...")
    # prepare features and labels
    categorical = ['PULocationID', 'DOLocationID']
    train_dict = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dict)
    y_train = df['duration'].values
    logger.info(f"Features shape: {X_train.shape}, Labels shape: {y_train.shape}")
    # train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    logger.info("Model training completed.")
    logger.info(f"Model intercept: {lr.intercept_}")
    return lr, dv


@task(log_prints=True)
def log_model():
    logger = get_run_logger()
    logger.info("Logging model to MLflow...")
    run_id = mlflow.active_run().info.run_id
    artifact_path = f"mlruns/0/{run_id}/artifacts/model"


@flow(log_prints=True)
def taxi_process_flow(url):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-prefect")
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        # Question 3
        _ = download_data(url)
        # Question 4
        df_filter = read_dataframe(url)
        # Question 5
        model, dv = train_model(df_filter)


if __name__ == "__main__":
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    taxi_process_flow(url)
