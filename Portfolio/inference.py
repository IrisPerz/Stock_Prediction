import joblib
import os
import pandas as pd
import json
import numpy as np
from io import BytesIO, StringIO

def model_fn(model_dir):
    path = os.path.join(model_dir, 'finalized_loan_model.joblib')
    model = joblib.load(path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
    elif request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    elif request_content_type == 'application/x-npy':
        data = np.load(BytesIO(request_body), allow_pickle=True)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_df, model):
    return model.predict(input_df)

def output_fn(prediction, content_type):
    res = prediction.tolist() if isinstance(prediction, (np.ndarray, np.generic)) else prediction
    return json.dumps(res), "application/json"
