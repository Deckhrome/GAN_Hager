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
        
        self.fc = nn.Linear(z_dim, 369 * 340 * 3)
        
        self.conv = nn.Conv2d(3, image_channels, kernel_size=3, stride=1, padding=1)
        
        self._init_weights()  
    def _init_weights(self):

        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.fc(x)  # Linear transformation to match image size
        x = x.view(-1, 3, 369, 340)  # Reshape to the desired image size
        x = torch.tanh(self.conv(x))  # Apply Conv2d and tanh activation
        return x



class Discriminator2(nn.Module):
    def __init__(self, image_channels, hidden_dim=32):
        super(Discriminator2, self).__init__()
        self.image_channels = image_channels
        self.conv1 = nn.Conv2d(image_channels, hidden_dim, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc_input_dim = hidden_dim * 183 *169 # Adjust this dimension to match the desired image size

        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Redimensionnement de la sortie pour l'entrée de la couche entièrement connectée
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Redimensionnement en déroulant les dimensions spatiales
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)