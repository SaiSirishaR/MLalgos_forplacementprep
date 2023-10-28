#### fitting the line to a new dataset from --> https://towardsdatascience.com/linear-regression-and-gradient-descent-for-absolute-beginners-eef9574eadb0
import numpy as np
import matplotlib.pyplot as plt


Ds = np.random.uniform(low=20, high=50, size = (10,2))
print("Ds is:", Ds)

x_inp = Ds[:,0]
y_inp = Ds[:,1]
plt.scatter(x_inp, y_inp)
plt.show()

## fit a line

m = 0
c= 0

epochs = 1000
L = 0.0001
N = len(x_inp)
print("N is", N)

for i in range(epochs): 
    Y_pred = m*x_inp + c  # The current predicted value of Y
    D_m = (-1/N) * sum(x_inp * (y_inp - Y_pred))  # Derivative wrt m
    D_c = (-1/N) * sum(y_inp - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)


Y_pred = m*x_inp + c

plt.scatter(x_inp, y_inp) 
plt.plot([min(x_inp), max(x_inp)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()
