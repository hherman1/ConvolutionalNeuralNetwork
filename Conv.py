from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer

from pyrbrain.structure.modules.neuronlayer import NeuronLayer

class ConvolutionalLayer(NeuronLayer):
	def _forwardImplementation(self,inbuf,outbuf):
		outbuf[:] = inbuf
	def _backwardImplementation(self,outerr,inerr,outbuf,inbuf):
		inerr[:] = outerr
	
