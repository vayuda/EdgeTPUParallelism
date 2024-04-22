import tensorflow as tf

def create_model():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the input data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

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

def trt_optimization():
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    # Instantiate the TF-TRT converter
    converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="mnist_model.keras",
    precision_mode=trt.TrtPrecisionMode.FP32
    )
    
    # Convert the model into TRT compatible segments
    trt_func = converter.convert()
    converter.summary()
    
    batch_size = 128
    def input_fn():
        batch_size = batch_size
        x = x_test[0:batch_size, :]
        yield [x]
        
        converter.build(input_fn=input_fn)
    
    OUTPUT_SAVED_MODEL_DIR="./models/tftrt_saved_model"
    converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)

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
    
model = create_model()
create_tflite_model(model)