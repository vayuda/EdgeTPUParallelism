import tensorflow as tf

def train_model(model):
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the input data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Save the trained model
    model.save("mnist_model.keras")
    print('saved model')
    return model

def load_model(path):
    model = tf.keras.models.load_model(path)
    print('loaded model')
    return model

def create_tflite_model(model):
    # Convert the saved model to TFLite format
    def representative_dataset():
        for _ in range(100):
            yield [tf.random.normal([1, 28, 28]),]

    quantizer = tf.lite.TFLiteConverter.from_keras_model(model)
    quantizer.optimizations = [tf.lite.Optimize.DEFAULT]
    quantizer.representative_dataset = representative_dataset
    quantizer.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quantizer.inference_input_type = tf.uint8  # or tf.uint8
    quantizer.inference_output_type = tf.uint8  # or tf.uint8
    
    tflite_quant_model = quantizer.convert()

    # Save the TFLite model to a file
    with open("mnist_model_batched.tflite", "wb") as f:
        f.write(tflite_quant_model)
    print("saved tf lite model")
    
