# import libraries
from huggingface_hub import hf_hub_url, hf_hub_download
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import requests
import torch
import cv2
import os

# Import functions from the module
from .rrdbnet_arch import RRDBNet
from .utils import *

# Models map to download automaticlly if not avilable
HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}

# Our main Uplcaling model handling class
class RealESRGAN:
    def __init__(self, device='cuda', scale=4):
        """
        RealESRGAN model that will handel all the upscaling and will help you with the image and video inference.
        args:
        1. device: Default=cuda but can be ['cuda' or 'cpu']
        2. scale: Default=4 can be [2, 4, 8] the upscaling factor 
        Usage example:
        model = RealESRGAN('cuda', 4)
        """
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=scale
        )
        
    # Get Model weights
    def load_weights(self, model_path, download=True):
        """
        Load the model weights into the model artitecture.
        Args:
        1. model_path: from where you want to load model, or where you want them to be saved.
        2. download: Default True. if the weights are not present do you want them to download them
        """
        # Check if the weights are present and download is enabled
        if not os.path.exists(model_path) and download:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            response = requests.get(config_file_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Weights downloaded to:", model_path)
            else:
                print("Error downloading:", response.status_code)
        
        # load the model weights
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        
        # load the model to cpu
        self.model.eval()
        self.model.to('cpu')
        
    def predict(self, lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15, mode='image'):
        """
        Upscale the image using this function by the factor that was passed in the constructor but the modl that was loaded.
        Args:
        1. lr_image: pillow format image that needs to be upscaled.
        2. batch_size: default=4
        3. patches_size: default=192
        4. padding: default=24
        5. pad_size: default=15
        6. mode: default 'image'
        => mode: don't chane this arg this decides weather to keep model loaded into cuda after inference if it is to upscale image
        that will unload the model to freeup space but if it is video mode the model will kept into cuda to speed up the inferencing process.
        
        Return: pillow formate image
        """
        
        # load the model to cuda
        if mode == 'image' and self.device == 'cuda':
            self.model.to(self.device)

        # extract parameters from the class
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        # split the image into patches
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )

        # load the patches to the device
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

        # Inference: Run the model to process the input patches
        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        # Post process the patches
        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        # Stitch the tiled/batched output into a single high-resolution image
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled, 
            target_shape=scaled_image_shape, padding_size=padding * scale
        )

        # Remove padding, scale pixel values to [0, 255] and make a PIL image
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)
        
        # unload the model to cpu if it was a image mode to freeup space
        if mode == 'image' and self.device == 'cuda':
            self.model.to('cpu')

        return sr_img
    
    def process_video(self, video_path, output_path, ffmpeg_bin='ffmpeg', batch_size=4, patches_size=192,padding=24, pad_size=15):
        """
        upscale the video using the RealESRGAN.
        Args:
        1. video_path: path to the video
        2. output_path: where do you want to save the video
        3. ffmpeg_bin: 'ffmpeg' if added to path or path to the ffmpeg bin
        4. batch_size: default=4
        5. patches_size: default=192
        6. padding: default=24
        7. pad_size: default=15

        Return: success(boolean), output_video_path
        """
        # success flag
        success = True

        # Extract full_path from the relative paths to avoid any errors
        video_path = os.path.abspath(video_path)
        output_path = os.path.abspath(output_path)
        print(video_name, output_path)


        # Make an output folder
        os.makedirs(output_path, exist_ok=True)
        print(f'Folder created at {output_path}')

        # Convert the video to MP4 if it is a FLV file
        if input_path.endswith('.flv'):
            mp4_path = input_path.replace('.flv', '.mp4')
            os.system(f'ffmpeg -i {input_path} -codec copy {mp4_path}')
            input_path = mp4_path

        # Extract the video name
        video_name = osp.splitext(os.path.basename(input_path))[0]
        video_save_path = osp.join(output_path, f'{video_name}_out.mp4')

        # Initialize the reader and writer
        reader = Reader(input_path, video_name, output_path, ffmpeg_bin)
        audio = reader.get_audio()
        height, width = reader.get_resolution()
        fps = reader.get_fps()
        writer = Writer(self.outscale, video_save_path, fps, width, height)
        print("Video Reading and Writing Initialized")

        # Load the model to cuda if avilabe:
        if self.device == 'cuda':
            self.model.to(self.device)
            print('model moved to cuda')

        # Process the video
        print('Start video inference...')
        pbar = tqdm(total=len(reader), unit='frame', desc='inference')
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                output = self.predict(img, batch_size, patches_size, padding, pad_size, mode='video')
            except RuntimeError as error:
                success = False
                break
            else:
                writer.write_frame(output)
            pbar.update(1)

        # unload the model from cuda
        if self.device == 'cuda':
            self.model.to('cpu')
            print('model moved back to cpu')

        # close the reader and writer to freeup space
        reader.close()
        writer.close()
        
        # Adding audio to the file
        """
        if success and reader.audio:
            os.system(f'ffmpeg -i "{input_path}" -q:a 0 -map a "{reader.tmp_frames_folder}/{audio.aac}" -y')
            os.system(f'ffmpeg -i "{video_save_path}" -i "{reader.tmp_frames_folder}/{audio.aac}" -c:v copy -c:a aac -strict experimental "{video_save_path.replace("_out.mp4", "_audio.mp4")}" -y')
            os.rename(video_save_path.replace("_out.mp4", "_audio.mp4"), video_save_path)
        """
        
        # Remove the temp fodler and the half processed files.
        if success:
            return success, video_save_path
        else:
            os.remove(video_save_path)
            return success, None

