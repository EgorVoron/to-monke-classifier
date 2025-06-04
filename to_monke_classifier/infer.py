import json

import numpy as np
import requests
from PIL import Image


def preprocess_image(img_file, img_size):
    img = Image.open(img_file).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # hwc -> chw
    arr = arr[np.newaxis, :]
    return arr.tolist()


def get_labels_dict(labels_info_path) -> dict:
    with open(labels_info_path, "r") as labels_info:
        return json.load(labels_info)


def run_inference(image_path, server_url, img_size, labels_info_path):
    input_arr = preprocess_image(image_path, img_size=img_size)
    data = json.dumps({"inputs": input_arr})
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, data=data, headers=headers)
    result = response.json()
    predictions = result.get("predictions")
    if not predictions:
        raise ValueError("no predictions")
    probs = np.array(predictions[0])
    label_id = int(np.argmax(probs))
    labels_info = get_labels_dict(labels_info_path)
    label_info = labels_info[str(label_id)]
    return (
        f"Your monkey's latin name is '{label_info['latin_name']}' "
        f"and common name is '{label_info['common_name']}'"
    )
