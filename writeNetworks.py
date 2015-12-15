# required imports
import sys

from time import clock
from os import mkdir
from os.path import exists

from CustomConv import SimpleConvolutionalNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.datasets.mnist import makeMnistDataSets
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import ModuleValidator
from pybrain.tools.shortcuts import buildNetwork

#Creates a directory if it does not exist
def guarenteeDir(path):
	if not exists(path):
		mkdir(path)

#Creates and writes a file containing the given string
def writeFile(path,string):
	f = open(path,'w')
	f.write(string)
	f.close()

#Builds the file structure for saving our results and networks
def buildStructure(root):
	base = root + "results/"
	guarenteeDir(base)
	guarenteeDir(lstmDir)
	guarenteeDir(convDir)
	guarenteeDir(ffDir)
	for (i,v) in enumerate(convolutionalTestValues):
		guarenteeDir(base + "convolutional/" + str(i))
	for (i,v) in enumerate(lstmTestValues):
		guarenteeDir(base + "lstm/" + str(i))
	for (i,v) in enumerate(feedForwardTestValues):
		guarenteeDir(base + "com/" + str(i))
	
#Make convolutional networks
def buildConvolutionalNetworks(base,tests):
	nets = []
	for (i,t) in enumerate(tests):
		start = clock()
		net = SimpleConvolutionalNetwork()
		net.genNetwork(1,28,12,4,t) #Defined in CustomConv, fills the network with t layers. 28 is the dimension of one side of our inputs "square", 12 is the amount to shrink that size by with each layer, and 4 is the number of subnodes in our tanh layer. 
		nets.append(net)
		end = clock()
		duration = end - start
		writeFile(base + str(i) + "/initalizeNetworkTime.txt","The run time to initialize the network was was:" + str(duration))
	return nets

#Make LSTM networks
def buildLSTMNetworks(base,tests):
	nets = []
	for (i,t) in enumerate(tests):
		start = clock()
		nets.append(buildNetwork(28*28,t*10,10,hiddenclass=LSTMLayer,recurrent=true) #input dimension is 28*28, hidden layer has t*10 nodes,output layer has 10 nodes.
		end = clock()
		duration = end - start
		writeFile(base + str(i) + "/initalizeNetworkTime.txt","The run time to initialize the network was was:" + str(duration))
	return nets
#Make FeedForward networks
def buildFeedForwardNetworks(base,tests):
	nets = []
	for (i,t) in enumerate(tests):
		start = clock()
		nets.append(buildNetwork(28*28,t*28,10) #input dimension is 28*28, hidden layer has t*28 nodes, output has 10 nodes
		end = clock()
		duration = end - start
		writeFile(base + str(i) + "/initalizeNetworkTime.txt","The run time to initialize the network was was:" + str(duration))
	return nets

#Train networks with backpropagation on MNIST
def trainNetworks(base,nets,ds):
	trainedNets = []
	for (i,n) in enumerate(nets):
		print "training",i
		start = clock()
		trainer = BackpropTrainer(n,ds)
		trainer.train()
		end = clock()
		duration = end - start
		NetworkWriter.writeToFile(trainer.module,base + str(i) + "/net.xml") #Save the network after training
		writeFile(base + str(i) + "/trainTime.txt","The network took " + str(duration) + "seconds to train.") #Record training time
		trainedNets.append(trainer.module)
		print i,duration #Tell you when one has finished training, since these things take a while.
	return trainedNets
#Evaluate the mean squared error of the network on the test set
def testAccuracy(base,nets,ts):
	for (i,n) in enumerate(nets):
		start = clock()
		mse = ModuleValidator.MSE(n,ts)
		end = clock()
		duration = end - start
		writeFile(base + str(i) + "/evaluationTime.txt","The network tested in " + str(duration) + " seconds with an MSE of " + str(mse))#record the MSE
		
if __name__ == "__main__":
	#define constants
	rootPath = "/Users/hherman1/ConvolutionalNeuralNetwork/"
	resDir = rootPath + "results/"
	convDir = resDir + "convolutional/"
	lstmDir = resDir + "lstm/"
	ffDir = resDir + "feedforward/"
	MNISTDIR = "MNIST/"
	#build training and testing data sets
	(train,test) = makeMnistDataSets(rootPath + MNISTDIR)
	
	#designate testing values
	convolutionalTestValues = [2,3,4]
	lstmTestValues = [1,2,5,10]
	feedForwardTestValues = [1,2,3,4,5,6]

	#build folder structure
	buildStructure(rootPath)
	
	#evaluate convolutional networks
	#RECCOMENDATION: Use the tensorflow implementation, also provided, or run no tests on the convolutional network, since these are exceptionally slow in pybrain.
	nets = buildConvolutionalNetworks(convDir,convolutionalTestValues)
	nets = trainNetworks(convDir,nets,train)
	testAccuracy(convDir,nets,test)

	#evaluate lstm networks
	nets = buildLSTMNetworks(lstmDir,convolutionalTestValues)
	nets = trainNetworks(lstmDir,nets,train)
	testAccuracy(lstmDir,nets,test)

	#evaluate FeedForward networks
	nets = buildFeedForwardNetworks(ffDir,convolutionalTestValues)
	nets = trainNetworks(ffDir,nets,train)
	testAccuracy(ffDir,nets,test)
