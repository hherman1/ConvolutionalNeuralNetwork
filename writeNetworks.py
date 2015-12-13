#generate xml neural networks and record the time taken to generate them jkj
import sys

from time import clock
from os import mkdir
from os.path import exists

from CustomConv import SimpleConvolutionalNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.datasets.mnist import makeMnistDataSets
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import ModuleValidator

def guarenteeDir(path):
	if not exists(path):
		mkdir(path)

def writeFile(path,string):
	f = open(path,'w')
	f.write(string)
	f.close()

def buildStructure(root):
	base = root + "results/"
	guarenteeDir(base)
	guarenteeDir(base + "convolutional/")
	guarenteeDir(base + "lstm/")
	guarenteeDir(base + "com/")
	for (i,v) in enumerate(convolutionalTestValues):
		guarenteeDir(base + "convolutional/" + str(i))
#	for (i,v) in enumerate(lstmTestValues):
#		guarenteeDir(base + "lstm/" + str(i))
	#for (i,v) in enumerate(comTestValues):
	#	guarenteeDir(base + "com/" + str(i))
	
#make convolutional networks
def buildConvolutionalNetworks(base,tests):
	nets = []
	for (i,t) in enumerate(tests):
		start = clock()
		net = SimpleConvolutionalNetwork()
		net.genNetwork(1,28,4,4,t)
		nets.append(net)
		end = clock()
		duration = end - start
		writeFile(base + str(i) + "/initalizeNetworkTime.txt","The run time to initialize the network was was:" + str(duration))
	return nets
def trainNetworks(base,nets,ds):
	trainedNets = []
	for (i,n) in enumerate(nets):
		start = clock()
		trainer = BackpropTrainer(n,ds)
		trainer.train()
		end = clock()
		duration = end - start
		NetworkWriter.writeToFile(trainer.module,base + str(i) + "/net.xml")
		writeFile(base + str(i) + "/trainTime.txt","The network took " + str(duration) + "seconds to train.")
		trainedNets.append(trainer.module)
#		print statement
		print i,duration
	return trainedNets
def testAccuracy(base,nets,ts):
	for (i,n) in enumerate(nets):
		start = clock()
		mse = ModuleValidator.MSE(n,ts)
		end = clock()
		duration = end - start
		writeFile(base + str(i) + "/evaluationTime.txt","The network tested in " + str(duration) + " seconds with an MSE of " + str(mse))
		
if __name__ == "__main__":
	rootPath = "/Users/hherman1/ConvolutionalNeuralNetwork/"
	resDir = rootPath + "results/"
	convDir = resDir + "convolutional/"
	MNISTDIR = "MNIST/"
	(train,test) = makeMnistDataSets(rootPath + MNISTDIR)

	convolutionalTestValues = [1,2,3,4]
	
	buildStructure(rootPath)
	nets = buildConvolutionalNetworks(convDir,convolutionalTestValues)
	nets = trainNetworks(convDir,nets,train)
	testAccuracy(convDir,nets,test)
