import torch as th
import numpy as np
import clip
import os
from PIL import Image

device = "cuda" if th.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load all images into a list here
img_names = ['/saltpool0/data/layneberry/WinoGround/WinoGroundPics/'+ line.strip() + '.png' for line in open('/saltpool0/data/layneberry/WinoGround/winoground_example_names.txt')]
img_list = [Image.open(fn) for fn in img_names]

# Load all texts into a list here
text_list = [line.strip() for line in open('/saltpool0/data/layneberry/WinoGround/winoground_captions.txt')]

images = th.stack([preprocess(img).to(device) for img in img_list])
texts = clip.tokenize(text_list).to(device)

print('images shape:', images.shape)
print('texts shape:', texts.shape)

with th.no_grad():
    # I think these are unnecessary?
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
    
    logits_per_image, logits_per_text = model(images, texts)
    print(logits_per_image.shape)
    input()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print('image feats shape', image_features.shape)
print('text feats shape', text_features.shape)
print('feats type', type(image_features), type(text_features))
print('probs shape', probs.shape)
print("Label probs:", probs)

np.save('/saltpool0/data/layneberry/WinoGround/clip_image_feats.npy', image_features.cpu().numpy())
np.save('/saltpool0/data/layneberry/WinoGround/clip_text_feats.npy', text_features.cpu().numpy())
np.save('/saltpool0/data/layneberry/WinoGround/clip_scores.npy', probs)
