# ----------------------------------------
# Imports
# ----------------------------------------
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from PIL import Image
import sys
 

if __name__ == "__main__":
    # ----------------------------------------
    # Save Results
    # ----------------------------------------
    results_path = "./results"
    # Create results folder if it doesn't exist

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plots_path = os.path.join(results_path, "./plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # ----------------------------------------
    # CSV
    # ----------------------------------------
    logs = open('stats.log', 'r')
    lines = logs.read().splitlines()
    logs.close()
    rows = []

    for line in lines:
        rows.append(json.loads(line.replace("\'", "\"")))

    df = pd.DataFrame(rows)


    # ----------------------------------------
    # Column stats
    # ----------------------------------------
    stats = {}
    for column in df:
        try:
            is_num = is_numeric_dtype(df[column][0])
            if is_num:
                stats[column] = {
                    "max": max(df[column]),
                    "min": min(df[column]),
                    "mean": np.mean(df[column]),
                    "median": np.median(df[column])
                }
        except:
            pass

    with open(os.path.join(results_path, "stats.json"), "w") as outfile:
        json.dump(stats, outfile)

    # ----------------------------------------
    # Plots
    # ----------------------------------------
    # CPUs
    for i in range(6):
        key = f"CPU{i + 1}"
        try:
            if key in df and is_numeric_dtype(df[key][0]):
                CPU = df[key]
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.set_title(key)
                ax.set_xlabel('Time')
                ax.set_ylabel('CPU Usage (%)')
                ax.plot(np.arange(len(CPU)), CPU)
                fig.savefig(plots_path + f"/{key}.png")
                plt.close(fig)
        except:
            pass

    # GPU
    GPU = df["GPU"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('GPU Progression')
    ax.set_xlabel('Time')
    ax.set_ylabel('GPU Usage (%)')
    ax.plot(np.arange(len(GPU)), GPU)
    fig.savefig(plots_path + '/GPU.png')
    plt.close(fig)

    # RAM
    RAM = df["RAM"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('RAM Progression')
    ax.set_xlabel('Time')
    ax.set_ylabel('RAM Usage (kB)')
    ax.plot(np.arange(len(RAM)), RAM)
    fig.savefig(plots_path + '/RAM.png')
    plt.close(fig)

    # Temperatures
    temps = ["AO", "AUX", "CPU", "GPU", "thermal", "PLL"]
    for temp in temps:
        key = f"Temp {temp}"
        if key in df:
            temp_values = df[key]
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.set_title(f"{temp} Temperature")
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature (Celcius)Z")
            ax.plot(np.arange(len(temp_values)), temp_values)
            fig.savefig(plots_path + '/temp_ao.png')
            fig.savefig(f"{plots_path}/{key.lower().replace(' ', '_')}")
            plt.close(fig)

    # Current Power
    power_cur = df["power cur"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Current Power')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Consumption (Milliwatt)')
    ax.plot(np.arange(len(power_cur)), power_cur)
    fig.savefig(plots_path + '/power_cur.png')
    plt.close(fig)

    # Average Power
    power_avg = df["power avg"]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('Average Power')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Consumption (Milliwatt)')
    ax.plot(np.arange(len(power_avg)), power_avg)
    fig.savefig(plots_path + '/power_avg.png')
    plt.close(fig)

