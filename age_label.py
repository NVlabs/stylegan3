import os
from time import perf_counter
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

import click
import numpy as np
import torch



#----------------------------------------------------------------------------

@click.command()
@click.option('--img_dir', help='Where to find images', type=str, default="out/images", required=False, metavar='DIR')
@click.option('--label_dir', help='Where to save labels', type=str, default="out/labels", required=False, metavar='DIR')
def age_label(
   img_dir: str,
   label_dir: str
):
    """Label images with pretrained model"""
    
    model_name = "nateraw/vit-age-classifier"
    print('Loading networks from "%s"...' % model_name)

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    os.makedirs(label_dir, exist_ok=True)

    # Labels.
    str_labels = ["0-2", "3-9" , "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    labels = []
    # Generate images.
    start_time = perf_counter()
    for img_idx, img_name in enumerate(os.listdir(img_dir)):
        inputs = extractor(images=Image.open(os.path.join(img_dir, img_name)), return_tensors="pt")
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        age = np.argmax(probs.detach().numpy(), axis=1)
        labels.append(age[0])
        if img_idx % 5 == 0:
            time_remaining = (perf_counter() - start_time) * (len(os.listdir(img_dir)) - img_idx) / (img_idx + 1)
            for i in range(len(str_labels)):
                print(f'{str_labels[i]}: {round(labels.count(i)/ len(labels) * 100, 4)}%')
            print('Time Remaining: %d' % (time_remaining))
            print("===================================")
            print("\033[11A", end="") # clear the screen


    # convert labels to tensor pt and save
    labels = torch.tensor(labels)
    torch.save(labels, label_dir+'/age_labels.pt')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    age_label() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
