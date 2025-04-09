# importing Libraries
from Real_ESRGAN.model import RealESRGAN
from PIL import Image
import torch

# Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and everything
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# input Image
path_to_image = 'input/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

# Infering the model
sr_image = model.predict(image)

# save the result
sr_image.save('output/sr_image.png')