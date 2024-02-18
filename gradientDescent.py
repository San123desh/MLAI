#Gradient descent for linear regression
#yhat = wx + b
#loss = (y - yhat) ** 2/N meansqerror/noofSamples
import numpy as np
#with x data and y data we can find what w and b holds
#training data
x = np.random.randn(10,1)
y = 9*x + np.random.rand()
#parameters
w = 0.0
b = 0.0
#hyperparameter
learning_rate = 0.01

#create gradient descent function
def descend(x,y,w,b,learning_rate):
    #initialise partial deritive
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    #zip allows to loop through both at same time
    for xi,yi in zip(x,y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    #make update with w parameter
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w,b
#iteratively make updates
for epoch in range(200):
    #run gradient descent
    w,b = descend(x,y,w,b,learning_rate)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2,axis = 0),x.shape[0])
    print(f"{epoch} loss is {loss},parameters w:{w}, b:{b}")