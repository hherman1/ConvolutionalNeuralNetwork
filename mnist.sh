rm -r MNIST
mkdir MNIST
curl "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" > MNIST/train-images-idx3-ubyte.gz
curl "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" > MNIST/train-labels-idx1-ubyte.gz
curl "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" > MNIST/t10k-images-idx3-ubyte.gz
curl "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" > MNIST/t10k-labels-idx1-ubyte.gz
gunzip MNIST/*
