# Projectile motion with drag using a "Physics informed neural network" (PiNN)

Physics informed neural networks (or PiNNs) are a use of neural networks in the realm of solving physics problems numerically. One is demonstrated here that finds the trajectory of a projectile with the drag force.  (Note: This "simple" system is not solvable analytically| for example using the kinematic equations you may know from an introductory physics class.)

Note: We know this problem is easily solved using a variety of well established techniques. We are using it here to learn about PiNNs.

Why PiNNs? Well, suppose you have a few data points, either from an experiment or some theory, but not enough data to train a neural network. Why? The data may be hard, expensive, or time-consuming to acquire. Or perhaps you have reached some other limitation in generating the data.  PiNNS do the following:

 1. Train a neural network on the data you have, and
 1. Use a differential equation ("the physics") to train the network on other areas of your domain for which you do not have data.

The outcome of training on data + physics, is sufficient (at least here) to find the trajectory of a projecile subject to the drag force.

# The System

## Data

Here, we pull projectile data from a known system that was solved numerically using simple Euler steps with a small time constant. The initial condition of the projectile were

 * $v$= 30 m/s
 * $\theta$ = $70^\circ$
 * $g$ = $9.8 m/s^2$
 * The combination of $v$ and $\theta$ gives
    * $v_{x0}$ = 10.26
    * $v_{y0}$ = 28.19
* With a drag cofficient $C=0.01$

The results of a numerical integration are [here](https://github.com/tbensky/PiNN_Projectile/blob/main/System/trajectory.csv), for which we plotted he $x$ and $y$ positions:

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


## Physics

The physics of drag is well known. We'll follow [this discussion](https://pubs.aip.org/aapt/pte/article-abstract/56/3/168/278226/When-Does-Air-Resistance-Become-Significant-in?redirectedFrom=fulltext), which organized Newton's Laws as follows:

$$F=Cv^2,$$

$$\frac{a_x}{dt}=-\frac{C}{m}vv_x,$$

and

$$\frac{a_y}{dt}=-g-\frac{C}{m}vv_y,$$

where $C$ is the drag coefficient, $g$ is gravity $(=9.8 m/s^2)$, and we'll put the mass of the projectile $m=1$.

## Goal


The goal here is to use a PiNN to draw a smooth curve through each data point, using only the 5 data points shown and the differential equations given for $a_x$ and $a_y$ (the x and y accelerations of the projectile).

# The neural network

We'll try this with a neural network that resembles this one:

![Figure 3](https://github.com/tbensky/PiNN_Projectile/blob/main/Media/diagrams/diagrams.001.jpeg).

Although the number of deep (or hidden layers) is left as an open variable (we actually used 2).

In other words, the single input to the network will be $t$ or time. This will feed one hidden layer, which will feed an output layer that will give us $x$, $y$, $v_x$ and $v_y$.  When a time value is input, we'd like the network to output the $(x,y)$ position of the projectile and the two components of its velocity, $(v_x,v_y)$.

## Pytorch

To begin, we'll set up the basic network

```python
class neural_net(nn.Module):
    def __init__(self):
        super(neural_net,self).__init__()
        self.input_neuron_count = 1
        self.hidden_neuron_count = 10
        self.output_neuron_count = 4

        #tanh works best for this
        self.activation = torch.nn.Tanh() 
        
        #2 layers seems to work
        self.layer1 = torch.nn.Linear(self.input_neuron_count, self.hidden_neuron_count)
        self.layer2 = torch.nn.Linear(self.hidden_neuron_count, self.output_neuron_count)

        #self.C = nn.Parameter(torch.rand(1), requires_grad=True)
        #self.C.clamp(0.01,1)


    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```


