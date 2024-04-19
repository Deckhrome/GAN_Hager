# prerequisites
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from generator import Generator 

# Create directories if they don't exist
os.makedirs('model_saved', exist_ok=True)
os.makedirs('samples', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir en tenseur PyTorch
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normaliser les valeurs
])

image_dir = "spectrogram"
train_dataset = ImageFolder(root=image_dir, transform=transform)

bs = 10
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)

z_dim = 1000
image_dim = 369 * 340 * 3  # Dimensions des images (hauteur x largeur x nombre de canaux)

# Définition du générateur et du discriminateur avec les bonnes dimensions

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
    def __init__(self, z_dim, image_channels, hidden_dim=64):
        super(Generator2, self).__init__()
        self.z_dim = z_dim
        self.image_channels = image_channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x.view(-1, self.z_dim, 1, 1))


class Discriminator2(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator2, self).__init__()
        self.image_channels = image_channels

        self.net = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim*4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

image_channels = 3  # Nombre de canaux pour les images RGB
#G = Generator2(z_dim=z_dim, image_channels=image_channels).to(device)
D = Discriminator2(image_channels=image_channels).to(device)

lr_G = 0.0002
lr_D = 0.0001
# Entraînement du modèle GAN
criterion = nn.BCELoss()
#G_optimizer = optim.Adam(G.parameters(), lr=lr_G)
D_optimizer = optim.Adam(D.parameters(), lr=lr_D)

def D_train(x):
    D.zero_grad()

    batch_size = x.size(0)  # Obtenir la taille du batch
    x_real, y_real = x.view(-1, image_dim), torch.ones(batch_size, 1)  # Utiliser batch_size pour obtenir le bon nombre de labels
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    z = torch.randn(batch_size, z_dim).to(device)  # Générer du bruit correspondant à la taille du batch
    x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    D_output = D(x_fake.detach())  # Détacher les sorties du générateur pour éviter le calcul du gradient à travers G
    D_fake_loss = criterion(D_output, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()  # Utiliser .item() pour extraire la valeur scalaire du loss tensor

def G_train():
    G.zero_grad()

    batch_size = x.size(0)  # Obtenir la taille du batch
    z = torch.randn(batch_size, z_dim).to(device)
    y = torch.ones(batch_size, 1).to(device)  # Utiliser batch_size pour obtenir le bon nombre de labels

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()  # Utiliser .item() pour extraire la valeur scalaire du loss tensor


# n_epoch = 11
# for epoch in range(1, n_epoch+1):
#     D_losses, G_losses = [], []
#     for batch_idx, (x, _) in enumerate(train_loader):
#         D_losses.append(D_train(x))
#         G_losses.append(G_train())

#     print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
#              epoch, n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    

# # Save
# torch.save(G.state_dict(), 'model_saved/generator_model.pth')
# torch.save(D.state_dict(), 'model_saved/discriminator_model.pth')
# Création d'une instance du générateur
G = Generator2(z_dim=z_dim, image_channels=image_channels)

# Définir la taille du batch et le nombre de canaux de l'image
batch_size = 1
image_channels = 3

# Générer du bruit aléatoire
z_dim = 1000  # Dimensions du vecteur de bruit
z = torch.randn(batch_size, z_dim).to(device)  # Générer du bruit correspondant à la taille du batch

# Utiliser le générateur pour produire une image
with torch.no_grad():
    generated_image = G(z)

# Vérifier les dimensions attendues
expected_dimensions = (batch_size, image_channels, 369, 340)
if generated_image.shape == torch.Size(expected_dimensions):
    print("La dimension de l'image générée est correcte.")
else:
    print("La dimension de l'image générée est incorrecte.")

# Détacher le tenseur pour éviter les problèmes avec numpy()
generated_image_np = generated_image.detach().cpu().numpy()

# Afficher l'image générée
generated_image_np = generated_image_np.squeeze(0)  # Supprimer la dimension de batch si nécessaire
plt.imshow(generated_image_np.transpose(1, 2, 0))  # Permuter les dimensions pour l'affichage RGB
plt.axis('off')
plt.title('Image générée')
plt.show()
