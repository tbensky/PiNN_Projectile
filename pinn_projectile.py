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
import time
import os


class neural_net(nn.Module):
    def __init__(self):
        super(neural_net,self).__init__()
        self.input_neuron_count = 1
        self.hidden_neuron_count = 25
        self.output_neuron_count = 4
        self.C = 0.01
        #self.C = nn.Parameter(torch.rand(1), requires_grad=True)

        #Both tanh and sigmoid should be tried as an activation
        self.activation = torch.nn.Tanh() 
        
        #3 layers seem to work OK
        self.layer1 = torch.nn.Linear(1,512)
        self.layer2 = torch.nn.Linear(512, 512)
        self.layer3 = torch.nn.Linear(512,512)
        self.layer4 = torch.nn.Linear(512,512)
        self.layer5 = torch.nn.Linear(512,512)
        self.layer6 = torch.nn.Linear(512,4)



    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.activation(x)
        
        x = self.layer3(x)
        x = self.activation(x)

        x = self.layer4(x)
        x = self.activation(x)

        x = self.layer5(x)
        x = self.activation(x)

        x = self.layer6(x)

        return x

    def getC(self):
        return self.C #self.C.item()

    def get_weight(self):
        return self.layer2.weight[3][3].item()

    def compute_ux(self,x_in):
        return torch.autograd.functional.jacobian(self, x_in, create_graph=True)

    def L(self,data,outputs,targets):
        #data_loss = torch.mean((outputs-targets)**2)
        #data_loss = torch.sqrt(data_loss)
        data_loss = torch.sqrt(torch.sum((outputs-targets)**2))
        #return data_loss

        phys_loss = 0.0
        g = 9.8

        #https://stackoverflow.com/questions/64988010/getting-the-outputs-grad-with-respect-to-the-input
        #https://discuss.pytorch.org/t/first-and-second-derivates-of-the-output-with-respect-to-the-input-inside-a-loss-function/99757
        #torch.tensor([t_raw],requires_grad = True)
        needed_domain = [torch.tensor([x],requires_grad=True) for x in [0.25,2.0,2.5,3.0,3.5,4.0,5.0,6.0]]
        #xl = [x/10.0 for x in range(4,40,1)]
        #needed_domain = [torch.tensor([x],requires_grad=True) for x in xl]

        for x_in in needed_domain:
            x_in = x_in.to(device)
            y_out = self.forward(x_in)

            #autograd.grad just sums gradients for a given layer
            #these didn't help us here
            #u_x = torch.autograd.grad(y_out, x_in, grad_outputs=torch.ones_like(y_out), create_graph=True, retain_graph=True)
            #u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x[0]), create_graph=True, retain_graph=True)
            
            
            u_x = self.compute_ux(x_in) #torch.autograd.functional.jacobian(self, x_in, create_graph=True) 
            u_xx = torch.autograd.functional.jacobian(self.compute_ux, x_in,create_graph=True)
        
            #compute the instantaenous speed
            vx = y_out[2]
            vy = y_out[3]
            v = torch.sqrt(vx*vx+vy*vy)
         
            #set the drag coefficient
            C =  self.C #self.get_weight() #self.getC() # 0.01 #self.get_weight()

            dx = C * v * vx
            dy = C * v * vy
            phys_loss += (u_xx[0] + dx)**2 + (u_xx[1] + g + dy)**2
      
        phys_loss = torch.sqrt(phys_loss)
        return data_loss + phys_loss

def find_speed():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu") and torch.backends.mps.is_available():
        device = torch.device("mps")
    device =  torch.device("cpu")
    return device

def dump_results(fcount,loss):
    ts = [x/10. for x in range(0,60,1)]
    x_nn = []
    y_nn = []

    with open("results.csv","w") as f:
        f.write("x,y,vx,vy,E\n")
        for t_raw in ts:
            t = torch.tensor([t_raw],requires_grad = True)
            t = t.to(device)
            y = ann.forward(t)
            vx = y[2]
            vy = y[3]
            h = y[1]
            vsq = vx*vx+vy*vy
            E = 0.5 * vsq + 9.8 * h
            f.write(f"{y[0].item()},{y[1].item()},{y[2].item()},{y[3].item()},{E}\n")

    x_data = []
    y_data = []
    for (input,output) in pairs:
        x_data.append(output[0])
        y_data.append(output[1])

    df = pd.read_csv("results.csv")
    plt.plot(df['x'],df['y'],color='blue')
    plt.xlim([0,40])
    plt.ylim([0,30])
    plt.plot(x_data,y_data,'o',color='orange')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"(phys+data), loss={loss:.2f}, C={ann.getC():.2f}")

    df = pd.read_csv("System/trajectory.csv")
    plt.plot(df['x'],df['y'],"g+")

    plt.savefig(f"Evolve/frame_{fcount:03d}.png",dpi=300)
    plt.close()


    df = pd.read_csv("loss.csv")
    plt.plot(df['epoch'],df['loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Projectile trajectory with drag - Loss vs. Epoch")
    plt.savefig("loss.png",dpi=300)
    #plt.draw()
    #plt.pause(0.01)


device = find_speed()
print(device)

#for best drag training: use 10-15 hidden_neuron_count for good training, lr=0.01
ann = neural_net()
ann.to(device)

optimizer = optim.SGD(ann.parameters(),lr=0.001,momentum=0.1)
#loss_fn = nn.MSELoss()

#projecile data with drag
#t,x,y,vx,vy data
pairs = [
    [[0.15],[1.505320908,4.024230274,9.81936043,25.49973351]],
    #[[0.5],[4.790417472,11.95657049,8.994238571,19.97982246]],
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
os.system("rm Evolve/*.png")
os.system("rm loss.csv")
with open("loss.csv","w") as f:
    f.write("epoch,loss\n")

es = time.time()
while True:
    loss_total = 0.0
    for (data,target) in train_loader:
        data, target = data.to(device), target.to(device)
        out = ann(data)
        loss = loss_fn(data,out,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()

    if epoch % 100 == 0:
        with open("loss.csv","a") as f:
            f.write(f"{epoch},{loss.item()}\n")

        ee = time.time()
        print(f"epoch={epoch},loss={loss_total}, {ee-es:.1f} sec")
        es = ee
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

ts = [x/10. for x in range(0,50,1)]
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
plt.plot(x_train,y_train,'.')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()


