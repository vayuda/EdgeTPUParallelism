import numpy as np
import tensorflow as tf
import time

for device in tf.config.list_physical_devices():
    print(device)
    
def generate_input():
    return np.random.rand(10, 28, 28).astype(np.float32)

def tflite_inference(num_trials):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_scale, output_zero_point = output_details[0]['quantization']
    input_scale, input_zero_point = input_details[0]["quantization"]
    
    # Prepare input data
    input_shape = input_details[0]['shape']
    inference_times = []
    
    for trial in range(num_trials):
        input_data = generate_input()
        start = time.perf_counter()
        input_data = (input_data / input_scale) + input_zero_point
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = output_scale * (output_data - output_zero_point)
        end = time.perf_counter()
        inference_times.append((end - start)*1000/10)

    avg_inference_time = np.mean(inference_times)
    print("average inference time: ", avg_inference_time, "ms")
    

def keras_inference(num_trials):
    # Load the Keras model
    model = tf.keras.models.load_model("mnist_model.keras")

    # Prepare input data
    input_shape = model.input_shape
    inference_times = []
    batch_size = 32
    for trial in range(num_trials):
        input_data = np.random.rand(batch_size, 28, 28).astype(np.float32)
        start = time.perf_counter()
        output_data = model.predict(input_data,verbose = 0)
        end = time.perf_counter()
        inference_times.append((end - start)*1000/batch_size)

    avg_inference_time = np.mean(inference_times)
    print("average inference time: ", avg_inference_time, "ms")

tflite_inference(100)