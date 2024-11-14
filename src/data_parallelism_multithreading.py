import logging
import threading
import time
import subprocess
import os
import csv

# running other file using run()
# model_type = "Dense"
model_type = "Alex"
def call_tpu(num_samples):
    if num_samples > 0:
        # logging.info("TPU Thread: starting")
        subprocess.run(["python", "tpu_data_parallelism.py", model_type, str(num_samples)])
        # logging.info("TPU Thread: finishing")
    else:
        with open("TPU_data_parallelism_times.txt", "a") as outfile_:
            outfile_.write("0\n")

def call_gpu(num_samples):
    if num_samples > 0:
        # logging.info("GPU Thread: starting")
        subprocess.run(["python", "gpu_data_parallelism.py", "GPU", model_type, str(num_samples)])
        # logging.info("GPU Thread: finishing")
    else:
        with open("GPU_data_parallelism_times.txt", "a") as outfile_:
            outfile_.write("0\n")

def call_cpu(num_samples):
    if num_samples > 0:
        # logging.info("CPU Thread: starting")
        subprocess.run(["python", "gpu_data_parallelism.py", "CPU", model_type, str(num_samples)])
        # logging.info("CPU Thread: finishing")
    else:
        with open("CPU_data_parallelism_times.txt", "a") as outfile_:
            outfile_.write("0\n")

if __name__ == "__main__":
    # https://realpython.com/intro-to-python-threading/
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("This function will output total time (including threading overhead) for each split of 10%, 20%, etc across devices. For individual device times, please consult the [device]_data_parallelism_times.txt document.")

    devices = ["TPU", "GPU", "CPU"]
    filenames = [os.path.join(prefix + "_data_parallelism_times.txt") for prefix in devices + ["multithreading",] ]
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)

    total_samples = 20
    step_size = 2
    splits = []
    for split_1 in range(0, total_samples + 1, step_size):
        for split_2 in range(0, total_samples - split_1 + 1, step_size):
            splits.append([split_1, split_2, total_samples - split_1 - split_2])
    # print(np.array(splits)/10)
    with open('multithreading_splits.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(splits)

    for split in splits:
        logging.info("Main: " + str(split))
        threads = []
        for i, func in enumerate((call_cpu, call_gpu, call_tpu)):
            threads.append(threading.Thread(target=func, args=(split[i],)))
        # logging.info("Main: before running threads")

        start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end = time.perf_counter()
        total_time = (end-start)*1000
        with open("multithreading_data_parallelism_times.txt", "a") as outfile:
            outfile.write(str(total_time) + "\n")
        time.sleep(3)
        # logging.info("Main: all threads done")
