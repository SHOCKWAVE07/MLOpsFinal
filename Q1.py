from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


processor = IrisDataProcessor()
processor.prepare_data()
print("Feature statistics:")
print(processor.get_feature_stats())
print("\nTraining data shape:", processor.X_train.shape)
print("Test data shape:", processor.X_test.shape)