import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
import json
import matplotlib.pyplot as plt
import math
import pandas as pd
import random

def find_speed():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu") and torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


class neural_net(nn.Module):
    def __init__(self):
        super(neural_net,self).__init__()
        self.input_neuron_count = 1
        self.hidden_neuron_count = 10
        self.output_neuron_count = 4

        #tanh works best for this
        self.activation = torch.nn.Tanh() 
        
        #6 layers seems about right
        self.layer1 = torch.nn.Linear(self.input_neuron_count, self.hidden_neuron_count)
        self.layer2 = torch.nn.Linear(self.hidden_neuron_count, self.output_neuron_count)

        self.C = nn.Parameter(torch.rand(1), requires_grad=True)
        #self.C.clamp(0.01,1)


    def forward(self,x):

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        return x

    def getC(self):
        return self.C.item()

    def clampC(self):
        self.C.clamp(0.01,1)
    
    def get_weight(self):
        return self.layer2.weight[3][3].item()

    def compute_ux(self,x_in):
        return torch.autograd.functional.jacobian(self, x_in, create_graph=True)

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
            #u_x = torch.autograd.grad(y_out, x_in, grad_outputs=torch.ones_like(y_out), create_graph=True, retain_graph=True)
            #print(u_x)
            #u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x[0]), create_graph=True, retain_graph=True)
            #print(u_xx)
            
            
            u_x = self.compute_ux(x_in) #torch.autograd.functional.jacobian(self, x_in, create_graph=True) 
            u_xx = torch.autograd.functional.jacobian(self.compute_ux, x_in,create_graph=True)
        
            vx = y_out[2]
            vy = y_out[3]
            v = torch.sqrt(vx*vx+vy*vy)
         
            #fit is through data points but very strange w/C=weight
            C =  0.01 #self.get_weight() #self.getC() # 0.01 #self.get_weight()

            dx = C * v * vx
            dy = C * v * vy
            phys_loss += (u_xx[0] + dx)**2 + (u_xx[1] + g + dy)**2
      
        phys_loss = torch.sqrt(phys_loss)
        #return data_loss
        #return phys_loss
        #return torch.add(data_loss,phys_loss)
        return data_loss + phys_loss

def dump_results(fcount,loss):
    ts = [x/10. for x in range(0,200,1)]
    x_nn = []
    y_nn = []

    with open("results.csv","w") as f:
        f.write("x,y\n")
        for t_raw in ts:
            t = torch.tensor([t_raw],requires_grad = True)
            y = ann.forward(t)
            f.write(f"{y[0].item()},{y[1].item()}\n")

    x_data = []
    y_data = []
    for (input,output) in pairs:
        x_data.append(output[0])
        y_data.append(output[1])

    df = pd.read_csv("results.csv")
    plt.clf()
    plt.plot(df['x'],df['y'])
    plt.plot(x_data,y_data,'o')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"(phys+data), loss={loss:.2f}, C={ann.get_weight():.2f}")
    #plt.savefig(f"Evolve/frame_{fcount:03d}.png",dpi=300)
    plt.draw()
    plt.pause(0.01)



device = find_speed()

#for best drag training: use 10-15 hidden_neuron_count for good training, lr=0.01
ann = neural_net()

optimizer = optim.SGD(ann.parameters(),lr=0.001)
#loss_fn = nn.MSELoss()

#projecile data with drag
#t,x,y,vx,vy data
pairs = [
    [[0.15],[1.505320908,4.024230274,9.81936043,25.49973351]],
    [[0.5],[4.790417472,11.95657049,8.994238571,19.97982246]],
    [[1.0],[9.068584899,20.24294006,8.176363485,13.35860991]],
    [[3.2],[25.11380976,23.69779604,6.604112216,-9.243313604]],
    [[4.7],[34.0088771,0.827816308,5.168032133,-20.42674118,]]
]

   



#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
inputs = []
targets = []
for (input,target) in pairs:
    inputs.append(input)
    targets.append(target)

inputs = torch.tensor(inputs,dtype=torch.float32,requires_grad=True)
target = torch.tensor(targets,dtype=torch.float32)

train = TensorDataset(inputs, target)
train_loader = DataLoader(train, batch_size=len(pairs), shuffle=False)

epoch = 0
loss_fn = ann.L
frame_count = 0
while True:
    loss_total = 0.0
    for (data,target) in train_loader:
        out = ann(data)
        loss = loss_fn(data,out,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()
        ann.clampC()

    if epoch % 100 == 0:
        print(f"epoch={epoch},loss={loss_total}")
        dump_results(frame_count,loss_total)
        frame_count += 1
        
    epoch += 1

    #need to train down to 1e-5 for ypp to work best
    if loss_total < 1e-3:
        break


x_train = []
y_train = []

for out in target:
    x_train.append(out[0])
    y_train.append(out[1])

ts = [x/10. for x in range(0,100,1)]
x_nn = []
y_nn = []
for t_raw in ts:
    t = torch.tensor([t_raw],requires_grad = True)
    y = ann.forward(t)
    x_nn.append(y[0].item())
    y_nn.append(y[1].item())

print(x_nn)
print(y_nn)
plt.plot(x_nn,y_nn,'.')
plt.plot(x_train,y_train,'o')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()


