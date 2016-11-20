"""

Example to create a neural network with one hidden layer for solving the classical XOR problem.

Based on the lua code at
https://github.com/torch/nn/blob/master/doc/training.md

@author: Naimish Agarwal

@dependencies: python 2.7, torch 7

"""


import lutorpy as lua

# set runtime options
lua.LuaRuntime(unpack_returned_tuples=True)

# import torch packages
lua.require("nn")
lua.require("torch")

# dataset is created to cater to the
# requirements of 'StochasticGradient' function in 'nn'
class Dataset(object):

    def __init__(self):

        # list of tuples
        self.data = []

        # generate training data with 3000 samples
        for i in range(0, 3000, 1):

            # 2 inputs
            x = torch.randn(2)

            # 1 output
            y = torch.Tensor(1)

            # calculate XOR
            if x[0] * x[1] > 0:
                y[0] = -1
            else:
                y[0] = 1

            self.data.append((i, x, y))

    def __getitem__(self, key):

        if key == "size":
            return lambda x: len(self.data)

        return self.data[key - 1]

dataset = Dataset()

# number of neuron units
num_input_units = 2
num_hidden_units = 20
num_output_units = 1

# multi-layer perceptron
mlp = nn.Sequential()

# add layers
mlp._add(nn.Linear(num_input_units, num_hidden_units))
mlp._add(nn.Tanh())
mlp._add(nn.Linear(num_hidden_units, num_output_units))

# loss criterion
loss_criterion = nn.MSECriterion()

# network trainer
trainer = nn.StochasticGradient(mlp, loss_criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 30
trainer.shuffleIndices = True
trainer.verbose = True
trainer._train(dataset)

# test the network
x = torch.Tensor(2)

x[0] = 0.5
x[1] = 0.5
print(mlp._forward(x))

x[0] = -0.5
x[1] = 0.5
print(mlp._forward(x))

x[0] = 0.5
x[1] = -0.5
print(mlp._forward(x))

x[0] = -0.5
x[1] = -0.5
print(mlp._forward(x))


"""

@output:

# StochasticGradient: training
# current error = 0.52182686250479
# current error = 0.35517207385762
# current error = 0.27558650734146
# current error = 0.23469956476742
# current error = 0.21860995787032
# current error = 0.20818956942104
# current error = 0.19910229929072
# current error = 0.1908585666759
# current error = 0.1833207562077
# current error = 0.1764517682122
# current error = 0.17058288536413
# current error = 0.16582304105578
# current error = 0.16190109477397
# current error = 0.1586126544871
# current error = 0.1558379800315
# current error = 0.15347669705696
# current error = 0.15144098859347
# current error = 0.14966010558804
# current error = 0.14807900405953
# current error = 0.14665417167925
# current error = 0.14535101676479
# current error = 0.14414399260266
# current error = 0.14301743762444
# current error = 0.14196374140675
# current error = 0.14097852567369
# current error = 0.14005664682161
# current error = 0.13919139668928
# current error = 0.1383755227382
# current error = 0.13760219061335
# current error = 0.13686533459795
# StochasticGradient: you have reached the maximum number of iterations
# training error = 0.13686533459795
-0.9656
[torch.DoubleTensor of size 1]

 0.9115
[torch.DoubleTensor of size 1]

 0.9545
[torch.DoubleTensor of size 1]

-1.0499
[torch.DoubleTensor of size 1]

"""
