import json
import os

def load_params(path="/home/vivekp22/Videos/vivek_Code_samples_practice/Boosten_House_pred_Tranformer/config/params.json"):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
