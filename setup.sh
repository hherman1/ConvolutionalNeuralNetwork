# env.sh installs virtualenv in a local directory if its not globally installed, builds a virtualenv, sources into it, and installs tensorflow/pybrain
sh env.sh
# mnist.sh installs the mnist dataset in ./MNIST/ and unzips it. Depends on the function gunzip
sh mnist.sh
