import numpy as np
import glob
import cv2


class Perceptron(object):
    #Constructor for the Perceptron neural-network
    def __init__(self):
        self.inputLayerSize = 16384
        self.outputLayerSize = 1
        self.W = np.random.randn(self.inputLayerSize,self.outputLayerSize)
        print(self.W.shape,"Weight shape")

    #Forward propagation
    #Sums all X.W and returns output
    def forward(self, X,):
        self.z = np.dot(X, self.W)
        yHat = self.sigmoid(self.z)
        #print(yHat.shape,"yhat shape")
        #print(self.z.shape,"self z shape")
        return yHat

    #Sigmoid activation function
    def sigmoid(self,z):
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

#Reads training data from the files, resizes all images to 128x128 and transforms it into a
#vector. Returns (Number_of_Training_Data , 128x128) np array
def getTrain_Input():
    cannons = np.array([cv2.resize(cv2.imread(file,0), (128,128)).flatten() for file in glob.glob("train/cannon/*.jpg")])
    cellphones = np.array([cv2.resize(cv2.imread(file,0),(128,128)).flatten() for file in glob.glob("train/cellphone/*.jpg")])
    return cannons, cellphones

#Sets expected output of the cannon to 0
#Sets expected output of the cellphone to 1
#cannons : Training cannon data, cellphones : Training cellphone data
#Returns (128x128, 1) np arrays
def setLayer(cannons,cellphones):
    cannonLayer = np.zeros((cannons.shape[0],1))
    cellphoneLayer = np.ones((cellphones.shape[0],1))
    return cannonLayer, cellphoneLayer

#Shuffles a and b in the same order.
#a : Training data, b: Training data expected outputs, lenght: Number of total training data
#Returns shuffled data
def randomize(a, b, lenght):
    permutation = np.random.permutation(lenght)
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

#Reads test input, sets images to NN input format
#Returns cannon and cellphone test data
def getTest_Input():
    cannon = np.array([cv2.resize(cv2.imread(file, 0), (128, 128)).flatten() for file in glob.glob("test/cannon/*.jpg")])
    cellphone = np.array([cv2.resize(cv2.imread(file, 0), (128, 128)).flatten() for file in glob.glob("test/cellphone/*.jpg")])
    return cannon, cellphone


cannon_train, cellphonne_train= getTrain_Input()
cannon_layer, cellphone_layer = setLayer(cannon_train, cellphonne_train)

#Combines Cannon Train data  and CellPhone Train data vertically
trainDataset = np.concatenate((cannon_train, cellphonne_train), axis=0)
#Combines Cannon Train labels  and CellPhone Train labels vertically
trainLabels = np.concatenate((cannon_layer, cellphone_layer), axis=0)
lenght = len(cannon_train)+len(cellphonne_train)
X, y = randomize(trainDataset, trainLabels, lenght)
print(f'X shape {X.shape}, y shape {y.shape}')

nn = Perceptron()
firstOutput = nn.forward(X)
np.savetxt('layers.txt', y, delimiter=',', fmt='%f')
np.savetxt('first_outputs.txt', firstOutput, delimiter=',', fmt='%f')
weights = nn.train(0.001, X, y, 1000)
yHat = nn.forward(X)
np.savetxt('output.txt', yHat, delimiter=',', fmt='%f')

correct=0
for layer, output in zip(y, yHat):
    if layer-output == 0:
        correct += 1

print(f'Correct guess :{correct} out of {lenght} training data accuracy %{100*(correct/lenght)}')

A, B = getTest_Input()
cannon = nn.forward(A)
cellphone = nn.forward(B)
print(cannon, "cannon must be 0")
print(cellphone, "cellphone must be 1")

np.savetxt('cannon_test_output.txt', cannon, delimiter=',', fmt='%f')
np.savetxt('cphone_test_output.txt', cellphone, delimiter=',', fmt='%f')
im_cannon = np.reshape(A[0],(128,128))
for i in range(len(A)-1):
    im_cannon = np.concatenate((im_cannon,np.reshape(A[i+1],(128,128))),axis=1)

im_cell = np.reshape(B[0],(128,128))
for i in range(len(B)-1):
    im_cell = np.concatenate((im_cell,np.reshape(B[i+1],(128,128))),axis=1)

im_hori = np.concatenate((im_cannon,im_cell), axis=0)
cv2.imshow("cellphones", im_hori)

cv2.waitKey(0)
cv2.destroyAllWindows()






