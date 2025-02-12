from pathlib import Path
import joblib

class PredictionPipeline:
    def __init__(self):
        self.preprocessor = joblib.load('artifacts/data_transformation/preprocessor.joblib')
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    
    def predict(self, data):
        data = self.preprocessor.transform(data)
        prediction = self.model.predict(data)

        return prediction