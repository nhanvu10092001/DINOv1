import os
import numpy as np
from PIL import Image

dummy_dir = os.path.join('data', 'train', 'dummy')
os.makedirs(dummy_dir, exist_ok=True)
for i in range(10, 65):
    im = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
    im.save(os.path.join(dummy_dir, f'{i}.jpg'))
print("Created extra dummy images.")
