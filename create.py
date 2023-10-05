import torch
from yolo import Model


def create_model_yolo():
    config = {
        "weight": "./weights/model.pt",
        "yaml": "./weights/data.yaml",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    createModel = Model(config["weight"], config["device"], config["yaml"])
    model = createModel.create()
    return model


def setup():
    model = create_model_yolo()
    return model
