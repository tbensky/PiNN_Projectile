import csv
import matplotlib.pyplot as plt
import pandas as pd


#df = pd.read_csv("trajectory.csv")
df = pd.read_csv("Outputs/results_666.csv")

plt.plot(df['t'],df['vx'],label="$v_x$")
plt.plot(df['t'],df['vy'],label="$v_y$")
plt.xlabel("Time (sec)")
plt.ylabel("$v_x$ or $v_y$ (m/s)")
plt.title("Projectile trajectory with drag - $v_x$ or $v_y$ vs time")
plt.legend()
plt.savefig("vxvy.png",dpi=300)
plt.show()


plt.plot(df['t'],df['x'],label="$x$")
plt.plot(df['t'],df['y'],label="$y$")
plt.xlabel("Time (sec)")
plt.ylabel("$x$ or $y$ (m)")
plt.title("Projectile trajectory with drag - $x$ or $y$ vs time")
plt.legend()
plt.savefig("xy.png",dpi=300)
plt.show()





