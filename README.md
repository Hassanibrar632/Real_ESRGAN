> **_NOTE:_**  The repo is created using this repo by [ai-forever](https://github.com/ai-forever/Real-ESRGAN)

# Real-ESRGAN
This repo is PyTorch implementation of a Real-ESRGAN model. This model shows better results on faces compared to the original version. The issue that were in the orignal repo are fixed and hopefully that will work.

> This is not an official implementation. I partially use code from the [ai-forever](https://github.com/ai-forever/Real-ESRGAN).

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images. 

- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Huggingface 🤗](https://huggingface.co/sberbank-ai/Real-ESRGAN)

### Installation

```bash
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
# we will get to that later this is not yet updated
```

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

### Examples

---

Low quality image:

![](input\lr_image.png)

Real-ESRGAN result:

![](output\sr_image.png)