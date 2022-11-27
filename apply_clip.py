import torch as th
import pickle as pkl
import clip
import os
from PIL import Image

winoground_path = '/saltpool0/data/layneberry/WinoGround/'

device = "cuda" if th.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load all images into a list here
img_names = [winoground_path+'WinoGroundPics/'+ line.strip() + '.png' for line in open(winoground_path+'winoground_example_names.txt')]
img_list = [Image.open(fn) for fn in img_names]

# Load all texts into a list here
text_list = [line.strip() for line in open(winoground_path+'winoground_captions.txt')]

images = th.stack([preprocess(img).to(device) for img in img_list])
texts = clip.tokenize(text_list).to(device)

with th.no_grad():
    logits_per_image, logits_per_text = model(images, texts)
    scores = logits_per_image / 100.

pkl.dump(scores.T.cpu(), open(winoground_path+'clip_scores.pkl','wb'))
