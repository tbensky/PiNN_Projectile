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
df = pd.read_csv("CSV/results_457.csv")
plt.plot(df['t'],df['vx'])
plt.xlabel("Time (sec)")
plt.ylabel("$v_x$")
plt.title("$v_x$ vs time")
plt.savefig("vx_vs_t.png",dpi=300)
plt.show()

