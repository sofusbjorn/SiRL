from autograd.number import Number
import jax

def test_autograd():

    x = Number(-4.0)
    x.dvdx = 1.0

    def test_func_process(x):
        z = 2 * x - 2 + x
        q = z.exp() + z * x
        h = z**2 + q
        y = h + q + q * x
        return y
    
    y = test_func_process(x)
    y.backward()
    x_impl, y_impl = x, y

    def test_func_proces_jax(x):
        z = 2 * x - 2 + x
        q = jax.numpy.exp(z) + z * x
        h = z**2 + q
        y = h + q + q * x
        return jax.numpy.reshape(y, ())
    

    x = jax.numpy.array([-4.0])
    y_jax = test_func_proces_jax(x)
    dydx = jax.grad(test_func_proces_jax)

    print(x_impl, y_impl)
    print(y_jax, dydx(x))

    # # forward pass went well
    # assert y_impl.val_ == ypt.data.item()
    # # backward pass went well
    # assert x_impl.dldv_ == dydx()

import numpy as np
import random
import matplotlib.pyplot as plt

def test_lincls():
    np.random.seed(0)
    random.seed(0)
    #test with linear classification
    #generate 2D data from two clusters
    N = 10
    pnts1 = np.random.multivariate_normal(mean=[1.5, 1.5], cov=[[0.5, -0.2], [-0.2, 0.5]], size=(N,))
    pnts2 = np.random.multivariate_normal(mean=[0.5, 0.5], cov=[[0.3, -0.1], [-0.1, 0.3]], size=(N,))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=pnts1[:, 0], y=pnts1[:, 1], c='blue')
    ax.scatter(x=pnts2[:, 0], y=pnts2[:, 1], c='orange')


    data = [(pnt, 0) for pnt in pnts1] + [(pnt, 1) for pnt in pnts2]
    random.shuffle(data)
    print([(pnt, 0) for pnt in pnts1] + [(pnt, 1) for pnt in pnts2])

    x1_data = [Number(val=d[0][0]) for d in data]
    x2_data = [Number(val=d[0][1]) for d in data]
    y_data = [Number(val=d[1]) for d in data]

    w1 = Number(val=random.gauss(0.0, 1.0))
    w2 = Number(val=random.gauss(0.0, 1.0))
    b = Number(val=0)

    def loss(w1, w2, b):
        tol_loss = Number(val=0.)
        for x1, x2, y in zip(x1_data, x2_data, y_data):
            y_hat = 1./((-(w1 * x1 + w2 * x2 + b)).exp() + 1)
            tol_loss = tol_loss + (y - y_hat) ** 2
        return tol_loss
    
    n_itr = 50
    lr = 1e-1
    for i in range(n_itr):
        l = loss(w1, w2, b)
        l.zero_grad()
        l.backward()
        #apply learning rate to w and b
        w1.val = w1.val - lr * w1.dldv
        w2.val = w2.val - lr * w2.dldv
        b.val = b.val - lr * b.dldv

        print("Iteration {0}: Loss - {1}".format(i, l.val))
    

    x1 = np.linspace(-1, 2, 50)
    x2 = (-b.val - w1.val * x1) / w2.val
    ax.plot(x1, x2, '-k', linewidth=8, )
    
    plt.show()
    return

def test_rosenbrock():
    np.random.seed(0)
    random.seed(0)
    def rosenbrock(x1, x2):
        a = 100
        b = 0
        return a*(x2-x1**2)**2 + (1-x1)**2 + b
    
    x1 = Number(val=random.gauss(1, 1))
    x2 = Number(val=random.gauss(0, 1))

    x1_array = np.linspace(-2, 2, 50)
    x2_array = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1_array, x2_array)    

    Z = rosenbrock(X1, X2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    surf = ax.plot_surface(X1, X2, Z, alpha=0.75)

    n_itr = 50
    lr = 1e-4
    for i in range(n_itr):
        l = rosenbrock(x1, x2)

        ax.plot([x1.val], [x2.val], [l.val], 'ro', markersize=12)

        l.zero_grad()
        l.backward()
                
        #apply learning rate to w and b
        x1.val = x1.val - lr * x1.dldv
        x2.val = x2.val - lr * x2.dldv

        print("Iteration {0}: Loss - {1}".format(i, l.val))

    
    plt.show()

if __name__ == "__main__":
    # test_autograd()
    # test_lincls()
    test_rosenbrock()