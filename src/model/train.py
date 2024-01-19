from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from typing import Tuple , Dict
from typing_extensions import Annotated
import pandas as pd
import joblib 

THIS_DIR = Path(__file__).parent
DIABETES_DEV_CSVPATH = (THIS_DIR  / "/Users/espoirbadohoun/RefactorDatascientistCode/experimentation/data/diabetes-dev.csv").resolve()
print(DIABETES_DEV_CSVPATH)




def load_data(path:str) -> pd.DataFrame :
     df = pd.read_csv(path)
     return df 



# Split the dataframe into test and train data
def split_data(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

      
    X = df.drop('Diabetic', axis=1).values
    y = df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    
    return X_train, X_test, y_train, y_test


# Train the model, return the model
def train_model(X_train:pd.DataFrame, y_train:pd.Series ,  args: Dict[str, float]) -> RegressorMixin:
    reg_model = Ridge(**args)
    reg_model.fit(X_train,  y_train)
    return reg_model


# Evaluate the metrics for the model
def get_model_metrics(reg_model:RegressorMixin, X_test:pd.DataFrame , y_test:pd.Series) -> Annotated[float, "r2_score"] :
    preds = reg_model.predict(X_test)
    mse = mean_squared_error(preds, y_test)
    metrics = {"mse": mse}
    return metrics


def main():
    # Start MLflow run
    with mlflow.start_run():

        # Load Data
        sample_data = load_data(DIABETES_DEV_CSVPATH)

        # Split Data into Training and Validation Sets
        X_train, X_test, y_train, y_test  = split_data(sample_data)

        # Train Model on Training Set
        args = {
            "alpha": 0.5
        }
        reg = train_model(X_train, y_train, args)

        # Validate Model on Validation Set
        metrics = get_model_metrics(reg, X_test , y_test)

        # Log parameters and metrics
        mlflow.log_params(args)
        mlflow.log_metrics(metrics)

        # Save Model
        model_name = "sklearn_regression_model.pkl"
        joblib.dump(value=reg, filename=model_name)

        # Log model artifact
        mlflow.sklearn.log_model(reg, "model")

if __name__ == '__main__':
    main()