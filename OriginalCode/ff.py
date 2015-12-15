__author__ = 'talhaahsan'
import pybrain
from pybrain.supervised.trainers import backprop
from pybrain.supervised.trainers import evolino
from pybrain.tools.datasets import mnist
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import ModuleValidator
import matplotlib.pyplot as plt
import time

FirstStartTime = time.time()
print("Making Networks 1-6")
net1 = buildNetwork(28*28, 28, 10)
net2 = buildNetwork(28*28, 28*2, 10)
net3 = buildNetwork(28*28, 28*3, 10)
net4 = buildNetwork(28*28, 28*4, 10)
net5 = buildNetwork(28*28, 28*5, 10)
net6 = buildNetwork(28*28, 28*6, 10)

#Gets the training and test sets for MNIST stuff
(train, test) = mnist.makeMnistDataSets("C:\\Users\\Talha Ahsan\\PycharmProjects\\PythonCrashCourse\\MNIST_data")
print("Training and Testing sets have been formed")
trainerrorlist = []
print("Starting backprop training")
timeTaken = []

starttime = time.time()
print("Training Network 1")
trainer1 = backprop.BackpropTrainer(net1, dataset= train)
score1 = trainer1.train()
trainerrorlist.append(score1)
timeTaken.append(time.time() - starttime)

starttime = time.time()
print("Training Network 2")
trainer2 = backprop.BackpropTrainer(net2, dataset= train)
score2 = trainer2.train()
trainerrorlist.append(score2)
timeTaken.append(time.time() - starttime)

starttime = time.time()
print("Training Network 3")
trainer3 = backprop.BackpropTrainer(net3, dataset= train)
score3 = trainer3.train()
trainerrorlist.append(score3)
timeTaken.append(time.time() - starttime)

starttime = time.time()
print("Training Network 4")
trainer4 = backprop.BackpropTrainer(net4, dataset= train)
score4 = trainer4.train()
trainerrorlist.append(score4)
timeTaken.append(time.time() - starttime)

starttime = time.time()
print("Training Network 5")
trainer5 = backprop.BackpropTrainer(net5, dataset= train)
score5 = trainer5.train()
trainerrorlist.append(score5)
timeTaken.append(time.time() - starttime)

starttime = time.time()
print("Training Network 6")
trainer6 = backprop.BackpropTrainer(net6, dataset= train)
score6 = trainer6.train()
trainerrorlist.append(score6)
timeTaken.append(time.time() - starttime)

print("BackPropTrainer 1 has an error rate of " + score1.__str__())
print("BackPropTrainer 2 has an error rate of " + score2.__str__())
print("BackPropTrainer 3 has an error rate of " + score3.__str__())
print("BackPropTrainer 4 has an error rate of " + score4.__str__())
print("BackPropTrainer 5 has an error rate of " + score5.__str__())
print("BackPropTrainer 6 has an error rate of " + score6.__str__())

testErrorArray = []
score1a = ModuleValidator.MSE(net1, test)
testErrorArray.append(score1a)
score2a = ModuleValidator.MSE(net2, test)
testErrorArray.append(score2a)
score3a = ModuleValidator.MSE(net3, test)
testErrorArray.append(score3a)
score4a = ModuleValidator.MSE(net4, test)
testErrorArray.append(score4a)
score5a = ModuleValidator.MSE(net5, test)
testErrorArray.append(score5a)
score6a = ModuleValidator.MSE(net6, test)
testErrorArray.append(score6a)

print("Printing mean square error values")
print("BackPropTrainer 1 has a test error rate of " + score1a.__str__())
print("BackPropTrainer 2 has a test error rate of " + score2a.__str__())
print("BackPropTrainer 3 has a test error rate of " + score3a.__str__())
print("BackPropTrainer 4 has a test error rate of " + score4a.__str__())
print("BackPropTrainer 5 has a test error rate of " + score5a.__str__())
print("BackPropTrainer 6 has a test error rate of " + score6a.__str__())

plt.plot([1,2,3,4,5,6], timeTaken)
plt.xlabel("Network")
plt.ylabel("Time to train on training set (s)")
plt.savefig("Time5.png")
plt.show()

print("--- %s seconds ---" % (time.time() - FirstStartTime))

