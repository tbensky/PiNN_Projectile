#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:28 2024

@author: tom
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:51:10 2024

@author: tom
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd


#df = pd.read_csv("trajectory.csv")
df = pd.read_csv("loss.csv")
plt.plot(df['epoch'],df['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Projectile trajectory with drag - Loss vs. Epoch")
plt.savefig("loss.png",dpi=300)
plt.show()

