#This is a modification of pybrain's built-in convolutional network class to allow the implementation of multiple stacked convolutional layers
#Each stacked convolutional layer is actually 2 pybrain modules, one is a tanh module with convLayerDim * secretNodes dimension, and the other is a sigmoid layer with convLayerDim dimension. 
#The primary work of this program is in appropriately connecting the neurons, which is a relatively difficult task in pybrain.  The primary modification we made
# was to stack more and more of these dual-module layers ontop of one another using pybrain's logic for connecting the layers appropriately. 

from __future__ import print_function

from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer

__author__ = 'Tom Schaul, tom@idsia.ch'

# TODO: code up a more general version
# TODO: use modulemash.viewonflatlayer()


class SimpleConvolutionalNetwork(FeedForwardNetwork):
    """ A network with a specific form of weight-sharing, on a single 2D layer,
    convoluting neighboring inputs (within a square). """

    def __init__(self, **args):
        FeedForwardNetwork.__init__(self, **args)

    def genNetwork(self, inputdim, insize, convSize, numFeatureMaps,numLayers):
        inlayer = LinearLayer(inputdim * insize * insize)
        self.addInputModule(inlayer)
        self._buildStructure(inputdim, insize, inlayer, convSize, numFeatureMaps,numLayers)
        self.sortModules()


    def _buildStructure(self, inputdim, insize, inlayer, convSize, numFeatureMaps,numLayers):
	prevLayer = inlayer
        outdim = insize

	for i in range(numLayers):
            prevLayer = self.addConvolutionalLayer(inputdim,outdim,prevLayer,convSize,numFeatureMaps)
            outdim = outdim - convSize + 1
        outLayer = SigmoidLayer(10)
        self.addOutputModule(outLayer)
        self.addConnection(FullConnection(prevLayer,outLayer))

    def addConvolutionalLayer(self, inputdim, insize, inlayer, convSize, numFeatureMaps):
	#build layers
        outdim = insize - convSize + 1
        hlayer = TanhLayer(outdim * outdim * numFeatureMaps, name='h')
        self.addModule(hlayer)

        outlayer = SigmoidLayer(outdim * outdim, name='out')
        self.addModule(outlayer)

        # build shared weights
        convConns = []
        for i in range(convSize):
            convConns.append(MotherConnection(convSize * numFeatureMaps * inputdim, name='conv' + str(i)))
        outConn = MotherConnection(numFeatureMaps)

        # establish the connections.
        for i in range(outdim):
            for j in range(outdim):
                offset = i * outdim + j
                outmod = ModuleSlice(hlayer, inSliceFrom=offset * numFeatureMaps, inSliceTo=(offset + 1) * numFeatureMaps,
                                     outSliceFrom=offset * numFeatureMaps, outSliceTo=(offset + 1) * numFeatureMaps)
                self.addConnection(SharedFullConnection(outConn, outmod, outlayer, outSliceFrom=offset, outSliceTo=offset + 1))

                for k, mc in enumerate(convConns):
                    offset = insize * (i + k) + j
                    inmod = ModuleSlice(inlayer, outSliceFrom=offset * inputdim, outSliceTo=offset * inputdim + convSize * inputdim)
                    self.addConnection(SharedFullConnection(mc, inmod, outmod))

	return outlayer	
