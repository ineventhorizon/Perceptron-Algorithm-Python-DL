import numpy as np
import glob
import cv2

#Perceptron NN Class
class Perceptron(object):
    #Constructor for the Perceptron neural-network
    def __init__(self):
        self.inputLayerSize = 16384
        self.outputLayerSize = 1
        self.W = np.random.randn(self.inputLayerSize,self.outputLayerSize)
        print(self.W.shape, 'Weight shape')

    #Forward propagation
    #Sums all X.W and returns output
    def forward(self, X):
        self.z = np.dot(X, self.W)
        yHat = self.sigmoid(self.z)
        return yHat

    #Sigmoid activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    #Derivative of the Sigmoid function
    def sigmoidPrime(self, z):
        return (self.sigmoid(z))*(1-self.sigmoid(z))

    #Trains the NN and updates self.W (Weights)
    #rate : Learning rate , X : Training data , y : Training labels (expected outputs)
    #iterNo : Iteration number
    def train(self, rate, X, y, iterNo):
        print(X.shape)
        itr = 0
        for iter in range(iterNo):
            print(f'Iteration : {itr}')
            self.yHat = self.forward(X)
            weight_change = rate*(y-self.yHat)*self.sigmoidPrime(self.yHat)
            deltaW = np.dot(weight_change.T, X).T
            self.W += deltaW
            itr += 1
        np.savetxt('weights.txt', self.W, delimiter=',', fmt='%f')
        return self.W


#Reads given data from the files, resizes all images to 128x128 and transforms it into a
#vector. Returns np array with shape (Number_of_Data , 128x128=16384)
def getInput(path):
    print(f'Reading {path} data')
    cannons = np.array([cv2.resize(cv2.imread(file, 0), (128, 128)).flatten() for file in glob.glob(f'{path}/cannon/*.jpg')])
    cellphones = np.array([cv2.resize(cv2.imread(file, 0), (128, 128)).flatten() for file in glob.glob(f'{path}/cellphone/*.jpg')])
    return cannons, cellphones

#Sets expected output of the cannon to 0
#Sets expected output of the cellphone to 1
#cannons : Training cannon data, cellphones : Training cellphone data
#Returns (128x128, 1) np arrays
def setLayer(cannons, cellphones):
    cannonLayer = np.zeros((cannons.shape[0], 1))
    cellphoneLayer = np.ones((cellphones.shape[0], 1))
    return cannonLayer, cellphoneLayer


#Shuffles a and b in the same order.
#a : Training data, b: Training data expected outputs, length: Number of total training data
#Returns shuffled data
def randomize(a, b, length):
    permutation = np.random.permutation(length)
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


cannon_train, cellphone_train = getInput('train')
cannon_layer, cellphone_layer = setLayer(cannon_train, cellphone_train)

#Combines Cannon Train data  and CellPhone Train data vertically
trainDataset = np.concatenate((cannon_train, cellphone_train), axis=0)
#Combines Cannon Train labels  and CellPhone Train labels vertically
trainLabels = np.concatenate((cannon_layer, cellphone_layer), axis=0)
length = len(cannon_train)+len(cellphone_train)
X, y = randomize(trainDataset, trainLabels, length)
print(f'X shape {X.shape}, y shape {y.shape}')

nn = Perceptron()
firstOutput = nn.forward(X)
np.savetxt('layers.txt', y, delimiter=',', fmt='%f')
np.savetxt('first_outputs.txt', firstOutput, delimiter=',', fmt='%f')
#Trains NN
weights = nn.train(0.001, X, y, 1000)
#Forward propagation with the new weights
yHat = nn.forward(X)
np.savetxt('output.txt', yHat, delimiter=',', fmt='%f')

#Finds correct guesses for training data and then calculates accuracy
correct = 0
for layer, output in zip(y, yHat):
    if layer-output == 0:
        correct += 1
print(f'Correct guess :{correct} out of {length} training data accuracy %{100*(correct/length)}')

#Testing
A, B = getInput('test')
cannon = nn.forward(A)
cellphone = nn.forward(B)
print(cannon, 'cannon must be 0')
print(cellphone, 'cellphone must be 1')

np.savetxt('cannon_test_output.txt', cannon, delimiter=',', fmt='%f')
np.savetxt('cphone_test_output.txt', cellphone, delimiter=',', fmt='%f')

#Displaying test images
#Top row = Cannons test data, Bottom row = Cellphone test data
#To display the test data vertically, number of CannonTestData 
#samples must be equal to CellphoneTestData
im_cannon = np.reshape(A[0], (128, 128))
for i in range(len(A)-1):
    im_cannon = np.concatenate((im_cannon, np.reshape(A[i+1], (128, 128))), axis=1)

im_cell = np.reshape(B[0], (128, 128))
for i in range(len(B)-1):
    im_cell = np.concatenate((im_cell, np.reshape(B[i+1], (128, 128))), axis=1)

im_hori = np.concatenate((im_cannon, im_cell), axis=0)

cv2.imshow('Test Data', im_hori)
cv2.waitKey(0)
cv2.destroyAllWindows()
