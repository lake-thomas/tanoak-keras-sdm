"""
Deep learning model templates for the NAIP + Climate SDM for Tan Oak.
Supports three models: 1) Climate-Only, 2) Image-Only, 3) Image + Climate

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RandomFlipVertical(layers.Layer):
    def call(self, inputs, training=None):
        if training:
            return tf.image.random_flip_up_down(inputs)
        return inputs


def build_image_backbone(image_size=256, dropout=0.25, augment=True):
    """
    4-band image encoder.

    Note:
    Keras applications usually expect 3 channels. To keep the code simple and robust,
    this version uses a small custom CNN that natively accepts 4-band NAIP imagery.
    That avoids awkward surgery on pretrained ImageNet weights.

    # To Do: Implement a small resnet image encoder (resnet18)
    """
    image_input = keras.Input(shape=(image_size, image_size, 4), name="image")
    x = image_input

    if augment:
        x = layers.RandomRotation(factor=0.15)(x)
        x = layers.RandomFlip(mode="horizontal")(x)
        x = RandomFlipVertical()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    return keras.Model(image_input, x, name="image_backbone")


def build_climate_mlp(num_env_features, dropout=0.25):
    env_input = keras.Input(shape=(num_env_features,), name="env")

    x = layers.Dense(256, activation="relu")(env_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    return keras.Model(env_input, x, name="climate_mlp")


def build_image_climate_model(num_env_features, image_size=256, dropout=0.25):
    image_encoder = build_image_backbone(image_size=image_size, dropout=dropout, augment=True)
    climate_encoder = build_climate_mlp(num_env_features=num_env_features, dropout=dropout)

    image_input = keras.Input(shape=(image_size, image_size, 4), name="image")
    env_input = keras.Input(shape=(num_env_features,), name="env")

    image_feat = image_encoder(image_input)
    env_feat = climate_encoder(env_input)

    x = layers.Concatenate()([image_feat, env_feat])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", name="probability")(x) # Output is binary classification, sigmoid

    return keras.Model(inputs=[image_input, env_input], outputs=output, name="tan_oak_image_climate_model")


def build_image_only_model(image_size=256, dropout=0.25):
    image_encoder = build_image_backbone(image_size=image_size, dropout=dropout, augment=True)
    image_input = keras.Input(shape=(image_size, image_size, 4), name="image")

    x = image_encoder(image_input)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", name="probability")(x) # Output is binary classification, sigmoid

    return keras.Model(inputs=image_input, outputs=output, name="tan_oak_image_only_model")


def build_climate_only_model(num_env_features, dropout=0.25):
    env_input = keras.Input(shape=(num_env_features,), name="env")

    x = layers.Dense(256, activation="relu")(env_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", name="probability")(x) # Output is binary classification, sigmoid

    return keras.Model(inputs=env_input, outputs=output, name="tan_oak_climate_only_model")


def build_model(model_type, num_env_features, image_size=256, dropout=0.25):
    if model_type == "image_climate":
        return build_image_climate_model(num_env_features, image_size=image_size, dropout=dropout)
    elif model_type == "image_only":
        return build_image_only_model(image_size=image_size, dropout=dropout)
    elif model_type == "climate_only":
        return build_climate_only_model(num_env_features, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")