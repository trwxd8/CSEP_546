import torch
import torch.nn.functional as func

class UpdatedBlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(UpdatedBlinkNeuralNetwork, self).__init__()
        
        torch.cuda.set_device(0)

        self.convLayerOne = torch.nn.Conv2d(1, 24, 6, 2).cuda()

        self.normLayerOne = torch.nn.BatchNorm2d(24).cuda()

        self.maxPoolingOne = torch.nn.MaxPool2d(kernel_size = 2, stride = 2).cuda()

        self.convLayerTwo = torch.nn.Conv2d(24, 64, 6, 2).cuda()

        self.normLayerTwo = torch.nn.BatchNorm2d(64).cuda()

        self.maxPoolingTwo = torch.nn.MaxPool2d(kernel_size = 2, stride = 2).cuda() 

        self.convLayerThree = torch.nn.Conv2d(64, 64, 2).cuda()


        self.convLayerFour = torch.nn.Conv2d(64, 80, 2).cuda()

        self.convLayerFive = torch.nn.Conv2d(80, 80, 2).cuda()


        self.maxPoolingThree = torch.nn.MaxPool2d(kernel_size = 2, stride = 1).cuda() 


        #self.convLayerFive = torch.nn.Conv2d(72, 72, 2).cuda()

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(80, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(5, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            ).cuda()

    def forward(self, x):
        #print("Input shape original:",x.shape)
        # Apply the layers created at initialization time in order
        out = func.relu(self.convLayerOne(x))
        #print("Input shape conv1:",out.shape)
        out = self.normLayerOne(out)
        out = self.maxPoolingOne(out)
        #print("Input pool 1:",out.shape)

        out = func.relu(self.convLayerTwo(out))
        #print("Input shape conv2:",out.shape)
        out = self.normLayerTwo(out)
        out = self.maxPoolingTwo(out)
        #print("Input pool 2:",out.shape)

        out = func.relu(self.convLayerThree(out))
        #print("Input shape conv3:",out.shape)

        out = func.relu(self.convLayerFour(out))
        #print("Input shape conv4:",out.shape)

        out = func.relu(self.convLayerFive(out))
        #print("Input shape conv5:",out.shape)

        out = self.maxPoolingThree(out)
        #print("Input pool 3:",out.shape)


        out = out.reshape(out.size(0), -1)
        #print("Input shape reshaped:",out.shape)

        out = func.relu(self.fullyConnectedOne(out))
        #print("Input shape fully 1:",out.shape)

        out = func.relu(self.fullyConnectedTwo(out))
        #print("Input shape fully 2:",out.shape)

        out = self.outputLayer(out)
        #print("Input shape output:",out.shape)

        return out

class NonWorkingBlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(UpdatedBlinkNeuralNetwork, self).__init__()
        
        torch.cuda.set_device(0)

        self.convLayerOne = torch.nn.Conv2d(1, 24, 8, 3).cuda()

        self.maxPoolingOne = torch.nn.MaxPool2d(kernel_size = 3, stride = 2).cuda()

        self.convLayerTwo = torch.nn.Conv2d(24, 72, 3).cuda()


        #self.maxPoolingOne = torch.nn.MaxPool2d(kernel_size = 3, stride = 2).cuda()
        self.maxPoolingTwo = torch.nn.MaxPool2d(kernel_size = 3, stride = 2).cuda() 

        self.convLayerThree = torch.nn.Conv2d(72, 85, 2).cuda()

        #self.maxPoolingThree = torch.nn.MaxPool2d(kernel_size = 3, stride = 1).cuda() 

        self.convLayerFour = torch.nn.Conv2d(85, 85, 2).cuda()
        #self.maxPoolingTwo = torch.nn.MaxPool2d(kernel_size = 3, stride = 2).cuda() 

        #self.convLayerFive = torch.nn.Conv2d(128, 85, 3)

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(2125, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(5, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            ).cuda()

    def forward(self, x):
        #print("Input shape original:",x.shape)
        # Apply the layers created at initialization time in order
        out = func.relu(self.convLayerOne(x))
        #print("Input shape conv1:",out.shape)
        out = self.maxPoolingOne(out)
        #print("Input pool 1:",out.shape)

        out = func.relu(self.convLayerTwo(out))
        #out = self.maxPoolingOne(out)
        #print("Input shape conv2:",out.shape)
        out = self.maxPoolingTwo(out)
        #print("Input pool 2:",out.shape)

        out = func.relu(self.convLayerThree(out))
        #print("Input shape conv3:",out.shape)
        #out = self.maxPoolingThree(out)

        out = func.relu(self.convLayerFour(out))
        #out = self.maxPoolingTwo(out)
       
        #print("Input shape conv4:",out.shape)

        #out = func.relu(self.convLayerFive(out))
        #print("Input shape conv5:",out.shape)

        out = out.reshape(out.size(0), -1)
        #print("Input shape reshaped:",out.shape)

        out = self.fullyConnectedOne(out)
        #print("Input shape fully 1:",out.shape)

        out = self.fullyConnectedTwo(out)
        #print("Input shape fully 2:",out.shape)

        out = self.outputLayer(out)
        #print("Input shape output:",out.shape)

        return out

class BestBlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(BestBlinkNeuralNetwork, self).__init__()
        
        self.convLayerOne = torch.nn.Conv2d(1, 6, 5).cuda()

        self.convLayerTwo = torch.nn.Conv2d(6, 16, 5).cuda()

        # Down sample the image to 12x12
        #self.avgPooling = torch.nn.AvgPool2d(kernel_size = 2, stride = 2) 

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(10000, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            ).cuda()

    def forward(self, x):
        #print("Input shape original:",x.shape)
        # Apply the layers created at initialization time in order
        out = func.max_pool2d(func.relu(self.convLayerOne(x)), (2,2))
        #print("Input shape conv1:",out.shape)
        out = func.max_pool2d(func.relu(self.convLayerTwo(out)), 2)
        #print("Input shape conv1:",out.shape)

        #out = self.avgPooling(out)
        #print("Input shape avg:",out.shape)

        out = out.reshape(out.size(0), -1)
        #print("Input shape reshaped:",out.shape)

        out = self.fullyConnectedOne(out)
        #print("Input shape fully:",out.shape)

        out = self.outputLayer(out)
        #print("Input shape output:",out.shape)

        return out

class ThreeLayerBlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(ThreeLayerBlinkNeuralNetwork, self).__init__()
        
        torch.cuda.set_device(0)

        self.convLayerOne = torch.nn.Conv2d(1, 32, 11, 4).cuda()

        self.maxPoolingOne = torch.nn.MaxPool2d(kernel_size = 3, stride = 2).cuda()

        self.convLayerTwo = torch.nn.Conv2d(32, 85, 5).cuda()

        self.maxPoolingTwo = torch.nn.MaxPool2d(kernel_size = 3, stride = 2).cuda() 

        self.convLayerThree = torch.nn.Conv2d(85, 85, 1).cuda()

        #self.maxPoolingThree = torch.nn.MaxPool2d(kernel_size = 3, stride = 1).cuda() 

        #self.convLayerFour = torch.nn.Conv2d(128, 128, 3)

        #self.convLayerFive = torch.nn.Conv2d(128, 85, 3)

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(765, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(5, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            ).cuda()

    def forward(self, x):
        #print("Input shape original:",x.shape)
        # Apply the layers created at initialization time in order
        out = func.relu(self.convLayerOne(x))
        #print("Input shape conv1:",out.shape)
        out = self.maxPoolingOne(out)
        #print("Input pool 1:",out.shape)

        out = func.relu(self.convLayerTwo(out))
        #print("Input shape conv2:",out.shape)
        out = self.maxPoolingTwo(out)
        #print("Input pool 2:",out.shape)

        out = func.relu(self.convLayerThree(out))
        #print("Input shape conv3:",out.shape)
        #out = self.maxPoolingThree(out)

        #out = func.relu(self.convLayerFour(out))
        #print("Input shape conv4:",out.shape)

        #out = func.relu(self.convLayerFive(out))
        #print("Input shape conv5:",out.shape)

        out = out.reshape(out.size(0), -1)
        #print("Input shape reshaped:",out.shape)

        out = self.fullyConnectedOne(out)
        #print("Input shape fully 1:",out.shape)

        out = self.fullyConnectedTwo(out)
        #print("Input shape fully 2:",out.shape)

        out = self.outputLayer(out)
        #print("Input shape output:",out.shape)

        return out

class BestModelBatchNorm(torch.nn.Module):
    def __init__(self, hiddenNodes = 20):
        super(BestModelBatchNorm, self).__init__()
        
        torch.cuda.set_device(0)

        self.convLayerOne = torch.nn.Conv2d(1, 32, 11, 3).cuda()

        self.normLayerOne = torch.nn.BatchNorm2d(32).cuda()

        self.maxPoolingOne = torch.nn.MaxPool2d(kernel_size = 2, stride = 2).cuda()

        self.convLayerTwo = torch.nn.Conv2d(32, 76, 3).cuda()

        self.normLayerTwo = torch.nn.BatchNorm2d(76).cuda()

        self.maxPoolingTwo = torch.nn.MaxPool2d(kernel_size = 2, stride = 2).cuda() 

        self.convLayerThree = torch.nn.Conv2d(76, 92, 2).cuda()

        #self.maxPoolingThree = torch.nn.MaxPool2d(kernel_size = 3, stride = 1).cuda() 

        #self.convLayerFour = torch.nn.Conv2d(64, 72, 2).cuda()

        #self.convLayerFive = torch.nn.Conv2d(72, 72, 2).cuda()

        # Fully connected layer to all the down-sampled pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(3312, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(5, hiddenNodes),
           torch.nn.Sigmoid()
           ).cuda()

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            ).cuda()

    def forward(self, x):
        #print("Input shape original:",x.shape)
        # Apply the layers created at initialization time in order
        out = func.relu(self.convLayerOne(x))
        #print("Input shape conv1:",out.shape)
        out = self.normLayerOne(out)
        out = self.maxPoolingOne(out)
        #print("Input pool 1:",out.shape)

        out = func.relu(self.convLayerTwo(out))
        #print("Input shape conv2:",out.shape)
        out = self.normLayerTwo(out)
        out = self.maxPoolingTwo(out)
        #print("Input pool 2:",out.shape)

        out = func.relu(self.convLayerThree(out))
        #print("Input shape conv3:",out.shape)
        #out = self.maxPoolingThree(out)

        #out = func.relu(self.convLayerFour(out))
        #print("Input shape conv4:",out.shape)

        #out = func.relu(self.convLayerFive(out))
        #print("Input shape conv5:",out.shape)

        out = out.reshape(out.size(0), -1)
        #print("Input shape reshaped:",out.shape)

        out = self.fullyConnectedOne(out)
        #print("Input shape fully 1:",out.shape)

        out = self.fullyConnectedTwo(out)
        #print("Input shape fully 2:",out.shape)

        out = self.outputLayer(out)
        #print("Input shape output:",out.shape)

        return out