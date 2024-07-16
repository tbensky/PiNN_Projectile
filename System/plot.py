#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:51:10 2024

@author: tom
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("trajectory.csv")
plt.plot(df['x'],df['y'])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Projectile trajectory with drag")
plt.legend()
plt.savefig("trajectory.jpg",dpi=300)
plt.show()

