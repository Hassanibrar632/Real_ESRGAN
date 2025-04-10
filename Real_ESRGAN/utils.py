from os import path as osp
import numpy as np
import shutil
import torch
import glob
import cv2
import os
import io

# Import ffmpeg
import ffmpeg

def pad_reflect(image, pad_size):
    imsize = image.shape
    height, width = imsize[:2]
    new_img = np.zeros([height+pad_size*2, width+pad_size*2, imsize[2]]).astype(np.uint8)
    new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image
    
    new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0) #top
    new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0) #bottom
    new_img[:, 0:pad_size, :] = np.flip(new_img[:, pad_size:pad_size*2, :], axis=1) #left
    new_img[:, -pad_size:, :] = np.flip(new_img[:, -pad_size*2:-pad_size, :], axis=1) #right
    
    return new_img

def unpad_image(image, pad_size):
    return image[pad_size:-pad_size, pad_size:-pad_size, :]


def process_array(image_array, expand=True):
    """ Process a 3-dimensional array into a scaled, 4 dimensional batch of size 1. """
    
    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch


def process_output(output_tensor):
    """ Transforms the 4-dimensional output tensor into a suitable image format. """
    
    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Pads image_patch with with padding_size edge values. """
    
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.
    The patches overlap by padding_size pixels.
    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.
    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size
    
    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    
    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    
    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)
    
    xmax, ymax, _ = padded_image.shape
    patches = []
    
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)
    
    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)
    
    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.
    After scaling, shapes and padding should be scaled too.
    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size
    
    complete_image = np.zeros((xmax, ymax, 3))
    
    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size,:
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]

# Important functions for video-streaming data handling
# All of these function are from the https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py

class Reader:
    def __init__(self, input_path, video_name, output_path, ffmpeg_bin):
        self.audio = None
        self.input_fps = None
        self.width = None
        self.height = None
        self.nb_frames = None
        self.idx = 0
        self.paths = []
        self.process_video(input_path, video_name, output_path, ffmpeg_bin)

    def process_video(self, video_path, video_name, output_path, ffmpeg_bin):
        """Extracts metadata and frames from the video."""
        # Get video metadata using ffmpeg
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

        # Extract metadata
        if video_stream:
            self.width = int(video_stream['width'])
            self.height = int(video_stream['height'])
            self.input_fps = eval(video_stream['r_frame_rate'])  # Convert "30/1" to float(30)
            self.nb_frames = int(video_stream.get('nb_frames', 0))  # Some formats may not have nb_frames
        if audio_stream:
            self.audio = audio_stream['codec_name']  # Store audio codec name

        # Extract frames from the video
        self.tmp_frames_folder = osp.join(output_path, f'{video_name}_inp_tmp_frames')
        os.makedirs(self.tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {video_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {self.tmp_frames_folder}/frame%08d.png')
        print(f"Extracting frames to: {self.tmp_frames_folder}")

        # Get all extracted frame paths
        self.paths = sorted(glob.glob(osp.join(self.tmp_frames_folder, "frame*.png")))
        self.nb_frames = len(self.paths)
        print(f"Extracted {self.nb_frames} frames from video.")

    def get_resolution(self):
        return self.height, self.width
    
    def get_fps(self):
        if self.input_fps is not None:
            return self.input_fps
        return 24
    
    def get_audio(self):
        return self.audio

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        return self.get_frame_from_list()
    
    def __len__(self):
        return self.nb_frames

    def close(self):
        shutil.rmtree(self.tmp_frames_folder)
        pass

class Writer:
    def __init__(self, outscale, output_path, fps, width, height):
        """Initialize OpenCV VideoWriter (without audio)."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        self.output_path = output_path
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (outscale*width, outscale*height))

    def write_frame(self, frame):
        """Write a single frame to the output video."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")

        if frame.dtype != np.uint8:
            raise ValueError("Frame data type must be np.uint8.")

        self.video_writer.write(frame)

    def close(self):
        """Release the video writer."""
        self.video_writer.release()

