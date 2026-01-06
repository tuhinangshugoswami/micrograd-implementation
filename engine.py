import math
class Value():
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev=set(_children)
        self.op=_op
        self.label=label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self,others):
        others = others if isinstance(others, Value) else Value(others)
        out= Value(self.data+others.data,(self,others),'+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            others.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self,others):
        others = others if isinstance(others, Value) else Value(others)
        out= Value(self.data*others.data,(self,others),'*')
        
        def _backward():
            self.grad += others.data * out.grad
            others.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
            
        out._backward = _backward
        return out
    
    def __neg__(self): 
        return self * -1

    def __sub__(self, other): 
        return self + (-other)
    
    def tanh(self):
        x= self.data
        t= math.tanh(x)
        out= Value(t,(self,),'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev: 
                    build_topo(child)
                topo.append(v)
        build_topo(self)
    
        self.grad = 1.0 
        for node in reversed(topo):
            node._backward()
    

