import random
import math
import copy
import json
import pickle

class network:
  def __init__(self,shape):
    self.shape = shape
    layers=[]
    for i in range(len(shape)-1):
      lyr=layer(shape[i],shape[i+1])
      layers.append(lyr)

    self.layers=layers
  
  def feed_forward(self,network_inputs):

    layer_outputs=self.layers[0].feed_forward(network_inputs)

    for l in range(1,len(self.layers)-1):
      layer_outputs=self.layers[l].feed_forward(layer_outputs)
    
    return self.layers[-1].feed_forward(layer_outputs,'linear')

  def birth(self,p):
    child = network(self.shape)
    #child.layers = copy.deepcopy(self.layers)

    for l,layer in enumerate(self.layers):
      for b,bias in enumerate(layer.biases):
        child.layers[l].biases[b] = lerp(bias, (random.random()*2)-1, p)

      for ow,out_weights in enumerate(layer.weights):
        for c,connection in enumerate(out_weights):
          child.layers[l].weights[ow][c] = lerp(
              connection, (random.random()*2)-1, p)
          
    return child
  
  def save(self,filename):
    with open(filename,'wb') as file:
      pickle.dump(self,file)

  def load(self,filename):
    with open(filename,'rb') as file:
      return pickle.load(file)

class layer:
  def __init__(self,inp,out):

   self.inp=inp
   self.out=out

   self.biases,self.weights=self.generate_layer_contents()
  
  def generate_layer_contents(self):

    bs=[]
    for i in range(self.out):
      bs.append((random.random()*2)-1)

    ws=[]
    for i in range(self.out):
      out_weights=[]
      for j in range(self.inp):
        out_weights.append((random.random()*2)-1)
      ws.append(out_weights)

    return bs,ws

  def feed_forward(self,layer_inputs,activation='relu'):
    layer_outputs=[]
    for j in range(self.out):
      act=0
      for i in range(self.inp):
        act+=(layer_inputs[i]*self.weights[j][i])

      if activation == 'relu':
        layer_outputs.append(relu(act-self.biases[j]))
      elif activation == 'sigmoid':
        layer_outputs.append(sigmoid(act-self.biases[j]))
      elif activation == 'linear':
        layer_outputs.append(act-self.biases[j])
      else:
        return None

    return layer_outputs

  
def sigmoid(x):
  return 1 - (1/(math.e**x + 1))

def relu(x):
  if x < 0:
    return 0
  
  return x

def lerp(a,b,t):
  return ((b-a)*t) + a


def test():
  net1 = network([3,5,5,1])
  net2 = network([1,5,5,1])
  net3 = net1.birth(0.01)
  net4 = net1.birth(1)
  net5 = net1.birth(0.01)
  
  print('random net 1 with input 1',net1.feed_forward([1,1,4]))
  print('random net 2 wiht input 1',net2.feed_forward([1]))
  print('net 3 birthed from 1 with 0.01',net3.feed_forward([1,1,4]))
  print('net 5 birthed from 1 with 0.01', net5.feed_forward([1, 1, 4]))
  print('net 4 birthed from 1 with 0.1', net4.feed_forward([1,1,4]))
  print('random net 1 (unmutated by birth?)', net1.feed_forward([1,1,4]))
  print('random net 1 one to one? ',net1.feed_forward([-1,0,1]))

  net1.save('./network_save.txt')
  net6 = network([3,5,1])
  print('pre load: ',net6.feed_forward([1,1,4]))
  net6 = net6.load('./network_save.txt')
  print('post load',net6.feed_forward([1,1,4]))


test()