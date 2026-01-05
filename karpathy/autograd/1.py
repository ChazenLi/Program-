class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        out = Value(-self.data, (self,), "neg")

        def _backward():
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = (2.0 / (1.0 + (2.718281828459045 ** (-2.0 * self.data)))) - 1.0
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = 2.718281828459045 ** self.data
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def example_simple():
    # y = x^2 + 2x + 1 at x=3 => y=16, dy/dx=2x+2=8
    x = Value(3.0)
    y = x * x + 2 * x + 1
    y.backward()
    print("simple:")
    print("y =", y)
    print("x =", x)


def example_chain():
    # A tiny MLP-like chain with tanh and exp
    x1 = Value(2.0)
    x2 = Value(-1.0)
    w1 = Value(0.5)
    w2 = Value(-1.5)
    b = Value(0.25)

    # z = x1*w1 + x2*w2 + b
    z = x1 * w1 + x2 * w2 + b
    # a = tanh(z), loss = exp(a)
    a = z.tanh()
    loss = a.exp()
    loss.backward()

    print("\nchain:")
    print("loss =", loss)
    print("x1 =", x1)
    print("x2 =", x2)
    print("w1 =", w1)
    print("w2 =", w2)
    print("b  =", b)


if __name__ == "__main__":
    example_simple()
    example_chain()
