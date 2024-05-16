import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator2(nn.Module):
    def __init__(self, z_dim, image_channels, hidden_dim=16):
        super(Generator, self).__init__()
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
        #on change la dimension de x en 3, 370, 340
        x= F.interpolate(x, size=(370, 340), mode='bilinear', align_corners=False)
        x = torch.tanh(self.conv(x))  # Apply Conv2d and tanh activation
        return x
    def summary(self):
        print(self)
        print("Nombre de paramètres : ", sum(p.numel() for p in self.parameters()))



class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.image_channels = image_channels
        self.conv1 = nn.Conv2d(image_channels, hidden_dim, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
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
    def summary(self):
        print(self)
        print("Nombre de paramètres : ", sum(p.numel() for p in self.parameters()))


class Generator(nn.Module):
    def __init__(self, z_dim, image_channels):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Premier redimensionnement: le vecteur z est mappé à une petite feature map
        self.fc = nn.Linear(z_dim, 256 * 23 * 21)  # Taille initiale ajustée pour le upscaling

        # Blocs de transposition convolutifs pour agrandir l'image
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(32)

        # Dernière couche pour atteindre la dimension des canaux d'image souhaitée
        # Ajustement des paramètres pour obtenir une taille proche de 369x340
        self.deconv4 = nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1, output_padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc(x)  # Transformation linéaire
        x = x.view(-1, 256, 23, 21)  # Redimensionnement en feature map

        x = F.relu(self.batchnorm1(self.deconv1(x)))  # Upsample et activation
        x = F.relu(self.batchnorm2(self.deconv2(x)))  # Upsample et activation
        x = F.relu(self.batchnorm3(self.deconv3(x)))  # Upsample et activation
        x = torch.tanh(self.deconv4(x))  # Dernier upsample et activation pour normalisation des pixels
        
        # Ajout d'un padding asymétrique pour obtenir exactement 369x340
        x = F.pad(x, (2, 3, 1, 1))  # Left, Right, Top, Bottom padding

        return x

    def summary(self):
        print(self)
        print("Nombre de paramètres : ", sum(p.numel() for p in self.parameters()))


