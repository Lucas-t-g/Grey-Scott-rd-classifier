import os
import logging
from time import sleep

# from threading import Thread
import matplotlib.pyplot as plt
from multiprocessing import Process

from demo__gray_scott_standard_model import generic, gs
from sim_folder_count import find_highest_simulation_number


def multiprocess_simulation(ratios, scales, seeds, sim_folder, step):
    logging.info(
        f"thread params: (ratios:{ratios}, scales:{scales}, seeds:{seeds}, sim folder:{sim_folder}, step:{step})"
    )
    for ratio in ratios:
        for scale in scales:
            for seed in seeds:
                diff_a, diff_b, delta_t, parms = ratio * scale, scale, 0.1, (0.098, 0.0555)
                params = {
                    "model": gs,
                    "delta_t": delta_t,
                    "diff_a": diff_a,
                    "diff_b": diff_b,
                    "parms": parms,
                    "stop": step,
                    "ini_a": 0.2,
                    "ini_b": 0.5,
                    "var_b": 0.4,
                    "info": True,
                    "size": 3,
                    "show": '',
                    "shape": 128,
                    "cmap": "gray",
                    "seed": seed,
                    # "pdf": f"{sim_folder}/ratio {ratio} seed {seed}.png",
                    "detect": True,
                }
                sim, a, b, c = generic(**params)
                # if not sim.is_uniform:
                plt.imsave(os.path.join(sim_folder, f"scale {scale} ratio {ratio} seed {seed}.png"), b, cmap="gray")

def process_are_working(process_list):#função que verifica se aainda tem threads trabalhando
    for process in process_list:
        # print("process is alive: ", process.is_alive())
        if ( process.is_alive() ):
            return True
    return  False

if __name__ == "__main__":
    sim_folder = f"simulations/simulation_{find_highest_simulation_number("./") + 1}"
    os.mkdir(sim_folder)
    # os.chdir(sim_folder)

    # ratio_range = 30
    # ratio_step = 1
    seed_range = 100
    seed_step = 1
    scale = 0.1
    step = 20000

    ratios = [4.5, 5, 6, 7, 8, 10, 15]
    scales = [0.1, 0.15, 0.2]
    seed = range(0, seed_range, seed_step)
    process_list = []

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Main    : before creating thread")

    for ratio in ratios:
        x = Process(target=multiprocess_simulation, args=([ratio], scales, seed, sim_folder, step))
        x.start()
        process_list.append(x)

    while (process_are_working(process_list)):
        sleep(1)

    print("END")


