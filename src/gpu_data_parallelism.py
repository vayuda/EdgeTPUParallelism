import numpy as np
import tensorflow as tf
import time
import os
import sys
from architectures import base_dnn, alexnet

cwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def benchmark(device_type, model_type, batch_size):

    model = base_dnn(5) if model_type == "Dense" else alexnet()

    inputshape = model.layers[0].input_shape[1:]
    device_name = '/gpu:0' if device_type == 'GPU' else '/cpu'

    with tf.device(device_name):
        times = []
        num_trials = 1
        for i in range(num_trials):
            start = time.perf_counter()
            outputs = model.predict(np.random.rand(batch_size, *inputshape), verbose=0)
            end = time.perf_counter()
            times.append((end-start)*1000)
        avg_inference_time = np.mean(times)
        with open(device_type + "_data_parallelism_times.txt", "a") as outfile:
            outfile.write(str(avg_inference_time) + "\n")

device_type = sys.argv[1]  # pass "GPU" or "CPU" as a command line parameter

model_type = sys.argv[2]

num_samples = int(sys.argv[3])

benchmark(device_type, model_type, num_samples)