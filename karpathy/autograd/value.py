import math

class Value:
    """存储标量值并支持自动微分的类"""
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # 梯度
        self._backward = lambda: None  # 反向传播函数
        self._prev = set(_children)  # 前驱节点
        self._op = _op  # 操作符，用于调试
        self.label = label  # 标签，用于调试

    def __add__(self, other):
        """加法运算"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """乘法运算"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """幂运算"""
        assert isinstance(other, (int, float)), "幂运算只支持数值类型"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        """右乘运算"""
        return self * other

    def __radd__(self, other):
        """右加运算"""
        return self + other

    def __truediv__(self, other):
        """除法运算"""
        return self * other**-1

    def __neg__(self):
        """负号运算"""
        return self * -1

    def exp(self):
        """指数函数"""
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # exp(x)的导数是exp(x)
        out._backward = _backward

        return out

    def tanh(self):
        """双曲正切函数"""
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """反向传播"""
        # 拓扑排序所有子节点
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # 从输出节点开始反向传播
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"