import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models import Generator2  # Import your Generator2 model definition

def generate_samples(model_path, output_path, z_dim=100, num_samples=5):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the Generator2 model
    generator = Generator2(z_dim=z_dim, image_channels=3).to(device)
    
    # Load the trained model weights
    generator.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model in evaluation mode (no gradient calculation during inference)
    generator.eval()
    
    # Generate and save sample images
    with torch.no_grad():
        # Generate random latent vectors for each sample
        z_samples = torch.randn(num_samples, z_dim, device=device)
        
        # Generate images using the generator model
        generated_images = generator(z_samples)
        
        # Save generated images as a grid
        save_image(generated_images, output_path, nrow=num_samples, normalize=True)

# Define paths and parameters
model_path = 'model_saved/generator_model.pth'  # Path to your saved generator model
output_path = 'samples/generated_samples.png'   # Output path for saving generated samples
num_samples = 1                                # Number of samples to generate in a row

# Generate and save sample images
generate_samples(model_path, output_path, z_dim=1000, num_samples=num_samples)
