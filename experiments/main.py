import numpy as np
import time
from experiments.learners import *
from experiments.tuned_learners import *
from data.data_to_use import *
import random

data = data_finnish()
repeats = 20


if __name__ == '__main__':

    list_CART = []

    time1 = time.time()
    for i in range(repeats):
        list_CART.append(CART_DE(data)[0])
    run_time1 = str(time.time() - time1)

    flat_list = np.array(list_CART).flatten()
    cart0_output = sorted(flat_list.tolist())

    print(cart0_output)
    print("median for CART0:", np.median(cart0_output))
    # print("mean for CART0:", np.mean(cart0_output))
    print("runtime for CART0:", run_time1)

    with open("./output/test_sk_mre.txt", "w") as output:
        output.write("CART0" + '\n')
        for i in sorted(cart0_output):
            output.write(str(i)+" ")
