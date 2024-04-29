# Lint as: python3

import numpy as np
import csv
import time
import os
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter


r"""Example using PyCoral to perform matrix multiplication

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
python examples/run_matmul.py
```
"""

input_dim = 64
output_dim = 64
max_val = 8
min_val = -8

cwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def scale_random(x):
    # scales input value from between [0,1) to between min_val and max_val
    return (x * (max_val - min_val)) + min_val

def generate_input(size=(1,input_dim)):
  x = scale_random(np.random.rand(*size))
  return x.astype(np.float32)

def main():


    for scaling in np.arange(1.0,7.1,.5):
        tflite_path = os.path.join(cwd, 'models', 'base_dnn', 'TPU_DNN_' + str(scaling) + ".tflite")

        # print(tflite_path)

        # # Load the TFLite model and allocate tensors.
        interpreter = make_interpreter(tflite_path)   # edge tpu specific interpreter
        interpreter.allocate_tensors()

        # Model must be uint8 quantized
        if common.input_details(interpreter, 'dtype') != np.uint8:
            raise ValueError('Only support uint8 input type.')

        # # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        output_scale, output_zero_point = output_details[0]['quantization']
        input_scale, input_zero_point = input_details[0]["quantization"]

        input_shape = tuple(input_details[0]['shape'])
        # print(input_shape)
        num_trials = 30
        inference_times = []
        for trial in range(num_trials):
            input_data = generate_input(input_shape)
            input_data = (input_data / input_scale) + input_zero_point
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
            start = time.perf_counter()
            interpreter.invoke()
            end = time.perf_counter()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = output_scale * (output_data - output_zero_point)
            inference_time = (end - start) * 1000
            inference_times.append(inference_time)

        avg_inference_time = np.mean(inference_times)
        print(avg_inference_time)



if __name__ == '__main__':
    main()
