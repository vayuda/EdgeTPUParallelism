import numpy as np
import tensorflow as tf
import time


def DNN_benchmark():
    from architectures import base_dnn
    model = base_dnn()
    batch_sizes = [1,4,16,64,256,1024]
    with tf.device('/gpu:0'):
        for batch_size in batch_sizes:
            times = []
            for i in range(30):
                start = time.perf_counter()
                outputs = model.predict(np.random.rand(batch_size, 64, 64), verbose=0)
                end = time.perf_counter()
                times.append((end-start)*1000)
            print(f"{np.mean(times):.3f}",end = " ")
        print("")

# check if gpu available before running
def gpu_benchmark(models, inputs):
    print(tf.config.experimental.list_physical_devices())
    with tf.device('/gpu:2'):
        batch_sizes = [1,4,16,64,256,512]
        for i,input_size in enumerate(inputs):
            model = tf.keras.models.load_model(f"{models}_{i+1}.keras")
            # print(f"GPU_CNN_{i} batch size 1, 4, 16, 256, 1024")
            print(f'gpu_{i+1}: ',end="")
            for batch_size in batch_sizes:
                times = []
                for i in range(30):
                    start = time.perf_counter()
                    outputs = model.predict(np.random.rand(batch_size, *input_size), verbose=0)
                    end = time.perf_counter()
                    times.append((end-start)*1000)
                print(f"{np.mean(times):.3f}",end = " ")
            print("")
            

def tflite_inference(num_trials=1000):
        
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="mnist_model_batched.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_scale, output_zero_point = output_details[0]['quantization']
    input_scale, input_zero_point = input_details[0]["quantization"]
    
    # Prepare input data
    input_shape = input_details[0]['shape']
    inference_times = []
    inputs = [tf.random.normal(None, *input_shape) for _ in range(num_trials)]
    for input_data in inputs:
        start = time.perf_counter()
        input_data = (input_data / input_scale) + input_zero_point
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = output_scale * (output_data - output_zero_point)
        end = time.perf_counter()
        inference_times.append((end - start)*1000)

    avg_inference_time = np.mean(inference_times)
    print("average inference time: ", avg_inference_time, "ms")

from architectures import layer_inputs
#conv1_inputs = layer_inputs["conv1"]
conv2_inputs = layer_inputs["conv2"]
# gpu_benchmark("models/conv1/GPU_CNN",conv1_inputs)
gpu_benchmark("models/conv2/GPU_CNN2",conv2_inputs)
'''
gpu_1: 77.467 67.594 81.479 139.446 394.105 706.269 
gpu_2: 68.076 64.920 78.002 124.814 294.545 504.879 
gpu_3: 68.462 63.382 73.136 124.063 295.556 506.752 
gpu_4: 63.590 66.613 72.559 120.937 292.907 507.013 
gpu_5: 64.318 74.434 123.207 291.897 976.743 1877.633 
gpu_6: 58.408 59.945 62.679 63.620 70.699 80.298 
gpu_7: 59.614 63.386 60.069 59.273 69.785 83.662
'''
# DNN_benchmark()
#85.304 68.740 61.173 70.153 98.186 210.742