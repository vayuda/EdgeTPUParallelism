import logging
import threading
import time
import subprocess

# running other file using run()

def call_tpu(name):
    logging.info("Thread %s: starting", name)
    subprocess.run(["python", "run_matmul.py"])
    logging.info("Thread %s: finishing", name)

def call_gpu(name):
    logging.info("Thread %s: starting", name)
    subprocess.run(["python", "gpu_shard.py"])
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    # https://realpython.com/intro-to-python-threading/
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,))
    y = threading.Thread(target=thread_function_2, args=(2,))
    logging.info("Main    : before running thread")
    x.start()
    y.start()
    logging.info("Main    : wait for the thread to finish")
    x.join()
    y.join()
    logging.info("Main    : all done")
