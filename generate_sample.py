import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models import Generator2  # Import your Generator2 model definition

def generate_samples(model_path, output_path, z_dim=1000, num_samples=1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator2(z_dim=z_dim, image_channels=3).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    with torch.no_grad():
        z_samples = torch.randn(num_samples, z_dim, device=device)
        generated_images = generator(z_samples)
        save_image(generated_images, output_path, nrow=num_samples, normalize=True)

model_path = 'model_saved/generator_model.pth' 
output_path = 'samples/generated_samples1.png'  
num_samples = 1              

generate_samples(model_path, output_path, num_samples=num_samples)
