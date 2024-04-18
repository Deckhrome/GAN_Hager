import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(g_input_dim, 128)
        self.fc2 = nn.Linear(128, g_output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = torch.tanh(self.fc2(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(d_input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.sigmoid(self.fc2(x))
        return x

class Generator2(nn.Module):
    def __init__(self, z_dim, image_channels, hidden_dim=16):
        super(Generator2, self).__init__()
        self.z_dim = z_dim
        self.image_channels = image_channels
        
        # Dimension after the linear layer
        self.fc = nn.Linear(z_dim, 369 * 340 * 3)  # Adjust this dimension to match the desired image size
        
        # Convolutional Transpose Layers
        self.ct1 = nn.ConvTranspose2d(3, hidden_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim * 4)
        
        self.ct2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        
        self.ct3 = nn.ConvTranspose2d(hidden_dim * 2, image_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        #print("Input shape:", x.shape)
        x = self.fc(x)
        #print("Shape after FC:", x.shape)
        x = x.view(-1, 3, 369, 340)  # Reshape to the desired image size
        #print("Shape after view:", x.shape)
        x = F.relu(self.bn1(self.ct1(x)))
        #print("Shape after ConvTranspose1 + BatchNorm1:", x.shape)
        x = F.relu(self.bn2(self.ct2(x)))
        #print("Shape after ConvTranspose2 + BatchNorm2:", x.shape)
        x = torch.tanh(self.ct3(x))  # Tanh for final output between -1 and 1
        #print("Final output shape:", x.shape)
        return x



class Discriminator2(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator2, self).__init__()
        self.image_channels = image_channels
        self.conv1 = nn.Conv2d(image_channels, hidden_dim, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc_input_dim = hidden_dim*2 * 90 * 83

        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Redimensionnement de la sortie pour l'entrée de la couche entièrement connectée
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Redimensionnement en déroulant les dimensions spatiales
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)