# Projectile motion with drag using a "Physics informed neural network" (PiNN)

Physics informed neural networks (or PiNNs) are a use of neural networks in the realm of solving physics problems numerically. One is demonstrated here that finds the trajectory of a projectile with the drag force.  (Note: This "simple" system is not solvable analytically (i.e. using the kinematic equations you may know from an introductory physics class.)

Why PiNNs? Well, suppose you have some data points, either from an experiment or some theory, but just a few. Why? Well, the data may be hard, expensive, or time-consuming to acquire. Or perhaps you are reaching the limits of your independent variables.  PiNNS to the following:

 1. Train a neural network on the data you have, and
 1. Use a differential equation to train the network on other domains of your system for which you do not have data.