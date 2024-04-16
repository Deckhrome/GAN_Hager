import torch
from torchvision.utils import save_image
from generator import Generator 

# Définir les paramètres nécessaires
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100  # Dimension de l'espace latent
image_dim = 369 * 340 * 3  # Dimensions de vos images

def generate_samples():
    G = Generator(g_input_dim=z_dim, g_output_dim=image_dim).to(device)
    G.load_state_dict(torch.load('model_saved/generator_model.pth', map_location=device))
    G.eval()

    with torch.no_grad():
        bs = 1
        z_sample = torch.randn(bs, z_dim).to(device)
        generated_images = G(z_sample).cpu().detach()

    save_image(generated_images, 'samples/generated_samples.png', nrow=5, normalize=True)



generate_samples()
