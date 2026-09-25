'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import argparse
import copy
import math
import os

import numpy as np
import yaml

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

DEFAULT_CONFIG = {
    "train": {
        "image_width": 150,
        "image_height": 150,
        "epochs": 10,
        "batch_size": 10,
        "data": {
            "train_dir": os.path.join("data", "train"),
            "validation_dir": os.path.join("data", "validation"),
        },
        "outputs": {
            "model_weights": "model.weights.h5",
            "train_features": "bottleneck_features_train.npy",
            "validation_features": "bottleneck_features_validation.npy",
            "metrics": "metrics.csv",
        },
    }
}


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as stream:
            user_config = yaml.safe_load(stream) or {}
        config = deep_update(config, user_config)
    return config["train"]


def directory_labels(data_dir):
    labels = []
    for class_index, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [
            name for name in sorted(os.listdir(class_dir))
            if os.path.isfile(os.path.join(class_dir, name))
        ]
        labels.extend([class_index] * len(files))
    return np.array(labels)


def save_bottleneck_features(config):
    image_size = (config["image_width"], config["image_height"])
    outputs = config["outputs"]
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        config["data"]["train_dir"],
        target_size=image_size,
        batch_size=config["batch_size"],
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict(
        generator, math.ceil(generator.samples / config["batch_size"]))
    np.save(open(outputs["train_features"], 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        config["data"]["validation_dir"],
        target_size=image_size,
        batch_size=config["batch_size"],
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict(
        generator, math.ceil(generator.samples / config["batch_size"]))
    np.save(open(outputs["validation_features"], 'wb'),
            bottleneck_features_validation)


def train_top_model(config):
    outputs = config["outputs"]
    train_data = np.load(open(outputs["train_features"], 'rb'))
    train_labels = directory_labels(config["data"]["train_dir"])

    validation_data = np.load(open(outputs["validation_features"], 'rb'))
    validation_labels = directory_labels(config["data"]["validation_dir"])

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=config["epochs"],
              batch_size=config["batch_size"],
              validation_data=(validation_data, validation_labels),
              verbose=0,
              callbacks=[TqdmCallback(), CSVLogger(outputs["metrics"])])
    model.save_weights(outputs["model_weights"])


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cats vs dogs model.")
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Path to the YAML training configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_config = load_config(args.config)
    save_bottleneck_features(train_config)
    train_top_model(train_config)
