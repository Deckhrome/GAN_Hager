import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import Generator, Discriminator
import time

# Création des répertoires s'ils n'existent pas
os.makedirs('model_saved', exist_ok=True)
os.makedirs('samples', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir en tenseur PyTorch
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normaliser les valeurs
])

image_dir = "spectrogram"
train_dataset = ImageFolder(root=image_dir, transform=transform)

batch_size = 10
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

z_dim = 1000  # Dimension du vecteur latent
image_dim = (3, 369, 340)  # Dimensions des images (nombre de canaux x hauteur x largeur)

# Initialisation du générateur et du discriminateur
G = Generator(z_dim=z_dim, image_channels=image_dim[0]).to(device)
D = Discriminator(image_channels=image_dim[0]).to(device)

G.summary()
D.summary()



lr_G = 0.0002
lr_D = 0.0001
# Entraînement du modèle GAN
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=lr_G)
D_optimizer = optim.Adam(D.parameters(), lr=lr_D)

# Démarrage du chronomètre
start_time = time.time()

def D_train(x):
    D.zero_grad()

    x_real = x.to(device)
    batch_size = x.size(0)
    y_real = torch.ones(batch_size, 1).to(device)

    #print("Dimension de x_real avant passage dans D2 :", x_real.shape)

    D_output_real = D(x_real)

    #print("Dimension de D_output_real :", D_output_real.shape)
    D_real_loss = criterion(D_output_real, y_real)

    z = torch.randn(batch_size, z_dim).to(device)
    x_fake = G(z)
    y_fake = torch.zeros(batch_size, 1).to(device)

    D_output_fake = D(x_fake.detach())
    D_fake_loss = criterion(D_output_fake, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def G_train(x):
    G.zero_grad()
    batch_size = x.size(0)

    z = torch.randn(batch_size, z_dim).to(device)
    G_output = G(z)
    D_output = D(G_output)
    y = torch.ones(batch_size, 1).to(device)
    G_loss = criterion(D_output, y)
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

num_epochs = 100
for epoch in range(1, num_epochs + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
             epoch, num_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    

# Arrêt du chronomètre et calcul du temps écoulé
end_time = time.time()
total_time = end_time - start_time

print(f"Entraînement terminé en {total_time:.2f} secondes.")
    
# plot loss
plt.figure()
plt.plot(D_losses, label='Discriminator loss')
plt.plot(G_losses, label='Generator loss')
plt.legend()
plt.savefig('loss.png')
plt.close()

# Sauvegarde des modèles entraînés
torch.save(G.state_dict(), 'model_saved/generator_model3.pth')
torch.save(D.state_dict(), 'model_saved/discriminator_model3.pth')
