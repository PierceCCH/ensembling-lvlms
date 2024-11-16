'''
 * The Recognize Anything Plus Model (RAM++) inference on unseen classes
 * Written by Xinyu Huang
'''
import os
import json
import torch
import numpy as np

from torch import nn
from PIL import Image

from ram import inference_ram_openset as inference
from ram import get_transform
from ram.models import ram_plus
from ram.utils import build_openset_llm_label_embedding


def generate_image_tags(folder_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    img_size = 384
    model_weights = "pretrained/ram_plus_swin_large_14m.pth"
    output_path = "tags.json"
    with open(output_path, 'w') as f:
        f.write('')

    transform = get_transform(image_size=img_size)

    model = ram_plus(pretrained=model_weights,
                    image_size=img_size,
                    vit='swin_l'
            )
    
    model.eval()
    model = model.to(device)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        tags = inference(img, model)
        
        with open(output_path, 'a') as f:
            json.dump({img_name: tags}, f)
            f.write('\n')

        print(img_name, ": ", tags)

    print('Tags generated and saved to', output_path)


folder_path = "test_images"
generate_image_tags(folder_path)