# Projectile motion with drag using a "Physics informed neural network" (PiNN)

Physics informed neural networks (or PiNNs) are a use of neural networks in the realm of solving physics problems numerically. One is demonstrated here that finds the trajectory of a projectile with the drag force.  (Note: This "simple" system is not solvable analytically, for example using the kinematic equations you may know from an introductory physics class.)

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

When numerically integrated, the results are found [here]()

