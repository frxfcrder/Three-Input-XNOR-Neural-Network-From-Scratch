import numpy as np

X = np.array(
    (
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ),
    dtype=float,
)

y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

xPredicted = np.array(([0, 0, 1]), dtype=float)

X = X / np.amax(X, axis=0)

xPredicted = xPredicted / np.amax(xPredicted, axis=0)


class NeuralNetwork:
    def __init__(self):
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def feedForward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.activationSigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)

        o = self.activationSigmoid(self.z3)
        return o

    def backwardPropagation(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.activationSigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):
        o = self.feedForward(X)
        self.backwardPropagation(X, y, o)

    def activationSigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def activationSigmoidPrime(self, s):
        return s * (1 - s)

    def saveSumSquaredLoss(self, i, error):
        with open("Sumsquaredlosslist.csv", "a") as loss_file:
            loss_file.write(f"{i},{error}\n")

    def saveWeights(self):
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self):
        print("Predicted XOR output data based on the weights: ")
        print("Expected (X1-X3): \n" + str(xPredicted))
        print("Output (Y1): \n" + str(self.feedForward(xPredicted)))


xorNetwork = NeuralNetwork()
trainingEpochs = 1000


with open("Sumsquaredlosslist.csv", "w") as lossFile:
    lossFile.write("Epoch,SumSquaredLoss\n")


for i in range(1, trainingEpochs + 1):
    print("Epoch #" + str(i) + "\n")
    print("Network Input: \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))
    print(
        "Actual Output of XOR Gate Neural Network: \n" + str(xorNetwork.feedForward(X))
    )
    Loss = np.sum(np.square(y - xorNetwork.feedForward(X)))
    xorNetwork.saveSumSquaredLoss(i, Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    xorNetwork.trainNetwork(X, y)

xorNetwork.saveWeights()
xorNetwork.predictOutput()
print(xorNetwork.feedForward(X))
