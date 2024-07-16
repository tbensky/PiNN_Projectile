import csv
import matplotlib.pyplot as plt
import pandas as pd
import math


pairs = [
    [[0.15],[1.505320908,4.024230274,9.81936043,25.49973351]],
    #[[0.3],[2.948769254,7.661012594,9.43427925,23.01909416]],
    [[0.5],[4.790417472,11.95657049,8.994238571,19.97982246]],
    [[1.0],[9.068584899,20.24294006,8.176363485,13.35860991]],
    [[3.2],[25.11380976,23.69779604,6.604112216,-9.243313604]],
    [[7.0],[34.0088771,0.827816308,2.864402605,-28.55554438]]
   
]

x_data = []
y_data = []
for (input,output) in pairs:
    x_data.append(output[0])
    y_data.append(output[1])


f = 0
while True:
    df = pd.read_csv("results.csv")
    plt.plot(df['x'],df['y'])
    plt.plot(x_data,y_data,'o')
    plt.savefig(f"Evolve/frame_{f:03d}.png",dpi=300)
    #plt.show()
