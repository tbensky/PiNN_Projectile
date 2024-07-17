# Projectile motion with drag using a "Physics informed neural network" (PiNN)

Physics informed neural networks (or PiNNs) are a use of neural networks in the realm of solving physics problems numerically. Here, we demonstrate this idea by using a neural network to find the trajectory of a projectile subjected to the drag force $Cv^2$.  (Notes: 1) This "simple" system is not solvable analytically, for example using the kinematic equations you may know from an introductory physics class, and 2) we know this problem is easily solved using a variety of well established techniques, but we still present it here as a learning exercise.)

Why PiNNs? Well, suppose in some study you're doing, you have *some* data, either from an experiment or some theory, but not enough data to train a neural network. Why? The data may be hard, expensive, or time-consuming to acquire. Or perhaps you have reached some other limitation in acquiring or generating the data.  PiNNS do the following:

 1. Train a neural network on the data you do have, and
 1. Use a differential equation ("the physics") to train the network on other areas of your domain for which you do not have data.

The outcome of training on data + physics, is sufficient (at least here) to find the trajectory of a projecile subject to the drag force.

# The System

## Data

Here, we pull projectile data from a known system that was solved numerically using Euler steps with a small time step. The initial condition of the projectile were:

 * $v$= 30 m/s
 * $\theta$ = $70^\circ$
 * $g$ = $9.8 m/s^2$
 * The combination of $v$ and $\theta$ gives
    * $v_{x0}$ = 10.26
    * $v_{y0}$ = 28.19
* A drag cofficient $C=0.01$

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

  Here is a plot of the $(x,y)$ points for each point:

  ![Figure 2](https://github.com/tbensky/PiNN_Projectile/blob/main/System/trajectory5.jpg?)


## Physics

The physics of drag is well known. We'll follow [this discussion](https://pubs.aip.org/aapt/pte/article-abstract/56/3/168/278226/When-Does-Air-Resistance-Become-Significant-in?redirectedFrom=fulltext), which organized Newton's Laws as follows:

$$F=Cv^2,$$

$$a_x=-\frac{C}{m}vv_x,$$

and

$$a_y=-g-\frac{C}{m}vv_y,$$

where $C$ is the drag coefficient, $g$ is gravity $(=9.8 m/s^2)$, and we'll put the mass of the projectile $m=1$.

## Goal


The goal here is to use a PiNN to draw a smooth curve through each data point, using only the 5 data points shown and the differential equations given for $a_x$ and $a_y$ (the x and y accelerations of the projectile).

# The neural network

We'll try this with a neural network that resembles this one:

![Figure 3](https://github.com/tbensky/PiNN_Projectile/blob/main/Media/diagrams/diagrams.001.jpeg).

Although the number of deep (or hidden layers) is left as an open variable (we actually used 2).

In other words, the single input to the network will be $t$ or time. This will feed one (or more) hidden layer(s), which will feed an output layer that will give us $x$, $y$, $v_x$ and $v_y$.  In other words, when a time value is input, we'd like the network to output the $(x,y)$ position of the projectile and the two components of its velocity, $(v_x,v_y)$.

## Pytorch


### Basic network structure
To begin, we'll set up the basic network in PyTorch

```python
class neural_net(nn.Module):
    def __init__(self):
        super(neural_net,self).__init__()
        self.input_neuron_count = 1
        self.hidden_neuron_count = 10
        self.output_neuron_count = 4

        #tanh and sigmoid should be tried with this
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

### The loss function
Next, we'll add the loss function. This is difference with PiNNs, becase they combine both data loss and physics loss, so you'll have to a custom loss function. We found with PyTorch, as long as it's a function within the `neural_network` class, the autodifferentiation seems to hold up. 

Here's the loss function we used (we left a bunch of comments in, to reflect things we learned and tried):

```python
def L(self,data,outputs,targets):
        data_loss = torch.mean((outputs-targets)**2)
        #data_loss = torch.sqrt(torch.sum((outputs-targets)**2))

        phys_loss = 0.0
        g = 9.8

        #https://stackoverflow.com/questions/64988010/getting-the-outputs-grad-with-respect-to-the-input
        #https://discuss.pytorch.org/t/first-and-second-derivates-of-the-output-with-respect-to-the-input-inside-a-loss-function/99757
        #torch.tensor([t_raw],requires_grad = True)
        needed_domain = [torch.tensor([x],requires_grad=True) for x in [0.25,2.0,6.0,8.0,10.0]]
        for x_in in needed_domain:
            y_out = self.forward(x_in)

            #autograd.grad just sums gradients for a given layer
            #these didn't help us here
            #u_x = torch.autograd.grad(y_out, x_in, grad_outputs=torch.ones_like(y_out), create_graph=True, retain_graph=True)
            #u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x[0]), create_graph=True, retain_graph=True)
            
            
            u_x = self.compute_ux(x_in)
            u_xx = torch.autograd.functional.jacobian(self.compute_ux, x_in,create_graph=True)
        
            #compute the instantaenous speed
            vx = y_out[2]
            vy = y_out[3]
            v = torch.sqrt(vx*vx+vy*vy)
         
            #set the drag coefficient
            C =  0.01 #self.get_weight() #self.getC() # 0.01 #self.get_weight()

            dx = C * v * vx
            dy = C * v * vy
            phys_loss += (u_xx[0] + dx)**2 + (u_xx[1] + g + dy)**2
      
        phys_loss = torch.sqrt(phys_loss)
        return data_loss + phys_loss
```

The first 3 lines of code are pretty standard: 1) compute the data loss, 2) initialize g (=9.8), and 3) initialize the physics loss variable.

Next, we build the `needed_domain` list, which reflects the domain for the training that our data doesn't support.  We'll take each domain point as an input to the network (in other words run each through the network on a forward pass), and then look at derivatives of the outputs produced.

We began computing the first and second derivatives using

```python
u_x = torch.autograd.grad(y_out, x_in, grad_outputs=torch.ones_like(y_out), create_graph=True, retain_graph=True)
u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x[0]), create_graph=True, retain_graph=True)
```

The Torch documentation seems to say that `.grad` will return a derivative vector of the same size as the `grad_outputs` vector. We were hoping to get a 4-component vector out $(v_x,v_y,a_x,a_y)$, so we used `torch.ones_like(y_out)` to make a ones vector which is the same size as the network output (4x1). But `.grad` still always returned just one number, which is the *sum* of the derivatives. So this returned $v_x+v_y+a_x+a_y$.

This, we had to define a function called `compute_ux` which is

```python
 def compute_ux(self,x_in):
        return torch.autograd.functional.jacobian(self, x_in, create_graph=True)
```

that computes the *Jacobian* of the network, which is strictly a vector of all possible derivatives of the network output.  To get the 2nd derivatives, we took a Jacobian of the first derivative. Thus, the pair of lines:

```python
u_x = self.compute_ux(x_in) 
u_xx = torch.autograd.functional.jacobian(self.compute_ux, x_in,create_graph=True)
```

seems to give us the first derivatives of the network output (`u_x`) and second derivative (`u_xx`).

Next, we compute the instantaneous speed of the projectile

```python
vx = y_out[2]
vy = y_out[3]
v = torch.sqrt(vx*vx+vy*vy)
```

Then we set the drag coefficient $C$ of

```python
C =  0.01 #self.get_weight() #self.getC()
```

We were hoping to allow the network to determine $C$. We tried a couple of approaches:

 1. Using an arbitrary weight of the network, `get_weight()`. This function looked like:
 ```python
     def get_weight(self):
        return self.layer2.weight[3][3].item()
```
 2. Using `getC()`, which is gets an additional trainable parameter we added to the network. We added this is the `___init()___` of the network class: `self.C = nn.Parameter(torch.rand(1), requires_grad=True)`.

 Neither of these techniques worked, so we just set $C$ to $0.01$, which was what was used to generate the numerical data. We are not sure why neither of these techniques work:

 1. Why can't we insist on a bit of additional constraint on a weight?
 1. Why won't a wholly trainable parameter of the network work?

Next, we computed the components of drag for both the $x$ and $y$ components to be

```python
dx = C * v * vx
dy = C * v * vy
```

and the loss just do to the physics
            
```python
phys_loss += (u_xx[0] + dx)**2 + (u_xx[1] + g + dy)**2
phys_loss = torch.sqrt(phys_loss)
```

then the total loss, which is returned by the loss function

```python
return data_loss + phys_loss
```


### Results

The $(x,y)$ output of the network was tracked as a function of training epoch.  In the plots shown, the big dots are the training data, the `+` symbols are the numerical integration results, and the solid curve is that from the neural network.  

Initially, the solid curve looks like this

![Figure 4](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_000.png?)

Note the small blue clump in the lower left corner. This is the sum total of the network's response.  As the epochs pass, the loss gets lower and lower and we'll get these:

![Figure 4](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_001.png?)

![Figure 5](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_002.png?)

![Figure 6](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_003.png?)

![Figure 7](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_004.png?)

![Figure 8](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_005.png?)

A few 1000s of epochs later, we see this appear

![Figure 9](https://github.com/tbensky/PiNN_Projectile/blob/main/Results/02/frame_011.png?)

which is about the best we've seen.
