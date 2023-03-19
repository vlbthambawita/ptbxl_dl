import torch.nn as nn

class XModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(XModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        return out

class Model(nn.Module):
    def __init__(self, X_input_dim, Z_input_dim, hidden_dim, dropout):
        super(Model, self).__init__()
        self.X_model = XModel(X_input_dim, hidden_dim, dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, Z_input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.X_model(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        out = self.sigmoid(out)
        return out

class CNN(nn.Module):
# Define the __init__ method that initializes the attributes of the class
    def __init__(self, X_input_dim, Z_input_dim, hidden_dim, dropout): 
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(X_input_dim, hidden_dim, kernel_size=3, padding=1) # first convolutional layer
        self.relu1 = nn.ReLU() # activation function (Rectified Linear Unit)
        self.dropout1 = nn.Dropout(dropout) # dropout layer
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1) # second convolutional layer
        self.relu2 = nn.ReLU() # activation function (Rectified Linear Unit)
        self.dropout2 = nn.Dropout(dropout) # dropout layer
        self.fc1 = nn.Linear(hidden_dim, 64) # first fully connected layer with hidden_dim input features and 64 output features
        self.fc2 = nn.Linear(64, 64) # second fully connected layer with 64 input features and 64 output features
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.out = nn.Linear(64, Z_input_dim) # output layer with 64 input features and 5 outputs corresponding to the diagnostic superclasses.
        self.sigmoid = nn.Sigmoid() # activation function for output layer (softmax)

    # forward method that performs the forward pass of the input data through the layers of the model
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = nn.functional.max_pool1d(out, kernel_size=out.size(-1)) # max pooling
        out = out.view(-1, self.conv2.out_channels)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        #out = self.sigmoid(out)
        return out
        
class CNN2(nn.Module):
    # Define the __init__ method that initializes the attributes of the class
    def __init__(self, input_dim, num_classes, hidden_dim, dropout, pooling=4): 
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1) # first convolutional layer
        self.relu1 = nn.ReLU() # activation function (Rectified Linear Unit)
        self.dropout1 = nn.Dropout(dropout) # dropout layer
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1) # second convolutional layer
        self.relu2 = nn.ReLU() # activation function (Rectified Linear Unit)
        self.dropout2 = nn.Dropout(dropout) # dropout layer
        self.pool = nn.MaxPool1d(pooling)
        self.fc1 = nn.Linear(hidden_dim * 1000 // pooling, 64) # first fully connected layer with hidden_dim input features and 64 output features
        self.fc2 = nn.Linear(64, 64) # second fully connected layer with 64 input features and 64 output features
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.out = nn.Linear(64, num_classes) # output layer with 64 input features and 5 outputs corresponding to the diagnostic superclasses.
        self.softmax = nn.Softmax(dim=1) # activation function for output layer (softmax)

    # forward method that performs the forward pass of the input data through the layers of the model
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        out = self.softmax(out)
        return out
