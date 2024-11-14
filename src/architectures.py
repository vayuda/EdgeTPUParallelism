import os.path

import tensorflow as tf
import numpy as np

def simple():
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def base_dnn(layer_scaling):
    # layer_scaling in [1, 2, ..., 7]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(int(2**layer_scaling), int(2**layer_scaling))),
        tf.keras.layers.Dense(int(2**(layer_scaling*1.5)), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(int(2**(layer_scaling*1.3)), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(int(2**(layer_scaling*1.1)), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(int(2**layer_scaling), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(int(2*layer_scaling), activation='softmax')
    ])
    return model


def alexnet():
    alexnet_model = tf.keras.models.Sequential([
        # https://medium.com/@syedsajjad62/alex-net-explanation-and-implementation-in-tensorflow-and-keras-8047efeb7a0f
        # https://www.wikiwand.com/en/AlexNet#Media/File:Comparison_image_neural_networks.svg
        # Layer 1: Convolutional layer with 64 filters of size 11x11x3
        tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                         input_shape=(227, 227, 3)),
        # Layer 2: Max pooling layer with pool size of 3x3
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
        tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer 6: Fully connected layer with 4096 neurons
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Layer 7: Fully connected layer with 4096 neurons
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Layer 8: Classification
        tf.keras.layers.Dense(1000, activation='softmax')
    ])
    return alexnet_model


def base_conv():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def conv2():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

layer_inputs= {
    "simple": (28, 28),
    "base_dnn": (28, 28),
    "base_conv": [(224,224,3),(224,224,3),(224,224,3),(224,224,3),(109, 109, 64), ( 51, 51, 128), (256,)],
    "conv2": [(128,128,3), (128,128,3),(128,128,3),(128,128,3), (61, 61, 64), (128, ), (512,)]
}


def convert_to_tflite(model, filename):
    input_size = model.layers[0].input_shape[1:]

    def representative_dataset():
        for _ in range(100):
            yield [np.random.rand(1, *input_size).astype(np.float32),]

    quantizer = tf.lite.TFLiteConverter.from_keras_model(model)
    quantizer.optimizations = [tf.lite.Optimize.DEFAULT]
    quantizer.representative_dataset = representative_dataset
    quantizer.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    quantizer.inference_input_type = tf.uint8
    quantizer.inference_output_type = tf.uint8

    tflite_quant_model = quantizer.convert()

    # Save the TFLite model to a file
    with open(filename, "wb") as f:
        f.write(tflite_quant_model)
    # print("saved tf lite model")
    return tflite_quant_model

# convert_to_tflite(alexnet(), os.path.join("models", "TPU_AlexNet.tflite"))
# convert_to_tflite(base_dnn(5), os.path.join("models", "TPU_DenseNet.tflite"))