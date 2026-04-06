import pickle
import numpy as np


def load_model(model_path):
    model = pickle.load(open(model_path, "rb"))
    return model


def predict(model, input_data):

    input_data = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_data)

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = None

    return prediction[0], probability
