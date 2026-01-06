import random
from engine import Value
class Neuron():
    def __init__(self,nin):
        self.w=[Value(random.uniform(-1,1))for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))

    def __call__(self,x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out= act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
class Layer():
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        params = []
        for n in self.neurons:
            ps = n.parameters()      
            params.extend(ps)
        return params
class MLP():
    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layer=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layer:
            x=layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layer:
                ps = layer.parameters()  
                params.extend(ps)
        return params
