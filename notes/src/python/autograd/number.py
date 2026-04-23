"""
A simple scalar system supporting auto-differentiation for a few arithmetic operations
Reference to Andrej Karpathy's micrograd implementation: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""

import math

class Number:
    def __init__(self, val) -> None:
        self.val = val
        self.dvdx = 0       #grad for forward mode
        self.dldv = 0       #grad for reverse mode
        self.prev = []
        self._backward = lambda: None   #do nothing by default
    
    def __add__(self, obj):
        #auto conversion when obj is float or int
        obj = obj if isinstance(obj, Number) else Number(val=obj)

        ret = Number(val=self.val + obj.val)
        ret.prev = [self, obj]

        #for forward mode
        # a + b = c
        # dcdx = dadx + dbdx
        ret.dvdx = self.dvdx + obj.dvdx

        #for reverse mode
        def _backward():
            # a + b = c
            # dlda = dldc * dcda = dldc
            self.dldv += ret.dldv
            obj.dldv += ret.dldv

        ret._backward = _backward

        return ret
    
    def __mul__(self, obj):
        #auto conversion when obj is float or int
        obj = obj if isinstance(obj, Number) else Number(val=obj)

        ret = Number(val=self.val*obj.val)
        ret.prev = [self, obj]

        #for forward mode
        # c = a * b
        # dcdx = dadx * b + a * dbdx
        ret.dvdx = self.dvdx * obj.val + self.val * obj.dvdx

        #for reverse mode
        def _backward():
            #a * b = c
            #dlda = dldc * dcda = dldc * b
            self.dldv += ret.dldv * obj.val
            obj.dldv += ret.dldv * self.val
        
        ret._backward = _backward

        return ret
    
    def __pow__(self, obj):
        # derivative dcda for a ** b = c requires a > 0
        # lets only accept constant obj
        assert isinstance(obj, (int, float))

        ret = Number(val=self.val ** obj)
        ret.prev = [self]

        #for forward mode
        # c = a ** b
        # dcdx = b * a ** (b-1) * dadx
        ret.dvdx = (obj * self.val**(obj-1)) * self.dvdx

        #for reverse mode
        def _backward():
            #a ** b = c
            #dlda = dldc * dcda = dldc * (b*a**(b-1))
            #dldb = dldc * dcdb = dldc * (a**b * log(a))
            self.dldv += ret.dldv * (obj * self.val**(obj-1))

        ret._backward = _backward
        return ret
    
    def exp(self):
        ret = Number(val=math.e ** self.val)
        ret.prev = [self]

        #for forward mode
        # c = e ** a
        # dcdx = e ** a * dadx
        ret.dvdx = math.e ** self.val * self.dvdx

        #for reverse mode
        def _backward():
            #a ** b = c
            #dlda = dldc * dcda = dldc * (b*a**(b-1))
            #dldb = dldc * dcdb = dldc * (a**b * log(a))
            self.dldv += ret.dldv * math.e ** self.val
        
        ret._backward = _backward

        return ret


    def __neg__(self):
        return self * Number(val=-1)
    
    def __radd__(self, obj):
        return self + obj
    
    def __sub__(self, obj):
        return self + (-obj)
    
    def __rsub__(self, obj): 
        return obj + (-self)

    def __rmul__(self, obj): 
        return self * obj

    def __truediv__(self, obj): 
        return self * obj**Number(val=-1)
    
    def __rtruediv__(self, obj): 
        return obj * self**-1
    
    def __repr__(self):
        return "Number(value={0}, fwd_grad={1}, bwd_grad={2})".format(self.val, self.dvdx, self.dldv)
    
    def _build_topo_ordered_set(self, topo_list=[], visited=set()):
        #traverse all number nodes that this number depends on by building a list subject to topological order
        #note there might be some nodes used multiple times, mark them to avoid multiple backward call
        if self not in visited:
            visited.add(self)
            for num in self.prev:
                num._build_topo_ordered_set(topo_list, visited=visited)
            topo_list.append(self)
        return 
        
    def backward(self):
        topo_ordered_list = []
        self._build_topo_ordered_set(topo_ordered_list, set())
        self.dldv = 1.  # l = v here
        for num in reversed(topo_ordered_list):
            num._backward()
        return
    
    def zero_grad(self):
        #reset dldv for all dependent nodes
        topo_ordered_list = []
        self._build_topo_ordered_set(topo_ordered_list, set())
        for num in reversed(topo_ordered_list):
            num.dldv = 0
        return
    

        

