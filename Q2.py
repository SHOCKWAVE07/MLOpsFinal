from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


class IrisDataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.features = None
        self.target = None
        self.scaled_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        iris = load_iris()
        self.data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],columns=iris['feature_names'] + ['target'])
        
        self.features = self.data.iloc[:, :-1]
        self.target = self.data['target']
        
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.features)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaled_features, self.target, test_size=self.test_size, random_state=self.random_state
        )
    
    def get_feature_stats(self):
        stats = pd.DataFrame(self.features.describe())
        return stats

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Ada Boost Classifier": AdaBoostClassifier()
        }
        self.results = {}

    def run_experiment(self):
        self.data_processor.prepare_data()

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(self.data_processor.X_train, self.data_processor.y_train)
                
                cv_accuracy = cross_val_score(model, self.data_processor.X_train, self.data_processor.y_train, cv=5, scoring="accuracy").mean()
    
                y_pred = model.predict(self.data_processor.X_test)
                accuracy = accuracy_score(self.data_processor.y_test, y_pred)
                precision = precision_score(self.data_processor.y_test, y_pred, average="weighted")
                recall = recall_score(self.data_processor.y_test, y_pred, average="weighted")
                
                self.results[model_name] = {
                    "CV Accuracy": cv_accuracy,
                    "Test Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall
                }
        
                self.log_results(model_name, cv_accuracy, accuracy, precision, recall, model)

    def log_results(self, model_name, cv_accuracy, accuracy, precision, recall, model):
        mlflow.log_metric("Cross-Validation Accuracy", cv_accuracy)
        mlflow.log_metric("Test Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
 
        mlflow.sklearn.log_model(model, model_name)

data_processor = IrisDataProcessor()
experiment = IrisExperiment(data_processor)
experiment.run_experiment()

for model_name, metrics in experiment.results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
