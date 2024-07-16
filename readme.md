# Projectile motion with drag using a "Physics informed neural network" (PiNN)

Physics informed neural networks (or PiNNs) are a use of neural networks in the realm of solving physics problems numerically. One is demonstrated here that finds the trajectory of a projectile with the drag force.  (Note: This "simple" system is not solvable analytically| for example using the kinematic equations you may know from an introductory physics class.)

Note: We know this problem is easily solved using a variety of well established techniques. We are using it here to learn about PiNNs.

Why PiNNs? Well, suppose you have a few data points, either from an experiment or some theory, but not enough data to train a neural network. Why? The data may be hard, expensive, or time-consuming to acquire. Or perhaps you have reached some other limitation in generating the data.  PiNNS do the following:

 1. Train a neural network on the data you have, and
 1. Use a differential equation ("the physics") to train the network on other areas of your domain for which you do not have data.

The outcome of training on data + physics, is sufficient (at least here) to find the trajectory of a projecile subject to the drag force.

# The system

Here, we pull projectile data from a known system that was solved numerically using simple Euler steps with a small time constant. The initial condition of the projectile were

 * $v$= 30 m/s
 * $\theta$ = $70^\circ$
 * $g$ = 9.8 m/$s^2$
 * The combination of $v$ and $\theta$ gives
    * $v_{x0}$ = 10.26
    * $v_{y0}$ = 28.19
* With a drag cofficient $C=0.01$

When numerically integrated, the results are found [here](https://github.com/tbensky/PiNN_Projectile/blob/main/System/trajectory.csv).  The $x$ and $y$ positions are graphed here:

![Figure 1](https://github.com/tbensky/PiNN_Projectile/blob/main/System/trajectory.jpg?)

Going with our idea that a PiNN is sparse data + physics, we'll remove all but 5 of the data points, which are as follows

  |     t     |     vx     |     vy     |     v     |     x     |     y     |  
  | --------- | ---------- |  --------- | --------- | --------- | --------- |
  |    0.15  |  9.81936043  |  25.49973351  |  27.3250114  |  1.505320908  |  4.024230274  |
  |    0.5  |  8.994238571  |  19.97982246  |  21.91094779  |  4.790417472  |  11.95657049  |
  |    1  |  8.176363485  |  13.35860991  |  15.66222776  |  9.068584899  |  20.24294006  | 
  |    3.2  |  6.604112216  |  -9.243313604  |  11.36015601  |  25.11380976  |  23.69779604  |
  |    4.7  |  5.168032133  |  -20.42674118  |  21.07036572  |  34.0088771  |  0.827816308  |

  Here is a plot of the 5 points:

  ![Figure 2](https://github.com/tbensky/PiNN_Projectile/blob/main/System/trajectory5.jpg?)


The goal here is to use a PiNN to draw a smooth curve through each data point.

