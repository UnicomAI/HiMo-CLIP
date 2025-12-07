import json
import cv2
from PIL import Image

from model import himo as longclip


import torch
import torch.utils.data as data
import os

image_root = 'data/Urban1k/image/'
caption_root_dict = {
    "en": 'data/Urban1k/caption/'
}

class local_dataset(data.Dataset):
    def __init__(self, image_root, caption_root):
        self.image_root = image_root
        self.caption_root = caption_root
        self.total_image = os.listdir(image_root)
        self.total_caption = os.listdir(caption_root)
    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        
        return image, caption

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1]
    model, preprocess = longclip.load(model_path, device=device)
    model.eval()
    print("model done!")
    
    language = sys.argv[2]
    caption_root = caption_root_dict[language]
    dataset = local_dataset(image_root=image_root, caption_root=caption_root)
    print(f"urban-1k, lang: {language}, model: {model_path}")

    img_feature_list = []
    text_list_1 = []
    text_list_2 = []
    text_list = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (image, caption) in enumerate(dataset):
            text_list.append(caption)

        text_feature = longclip.tokenize(text_list, truncate=True).to(device)
        text_feature = model.encode_text(text_feature)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        for i, (image, caption) in enumerate(dataset):            
            image = preprocess(image).unsqueeze(0).to(device)
            if i == 0:
                print(f"image: {image.shape}")
            img_feature = model.encode_image(image)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        print("image to text")
        i = 0
        correct = 0
        total = 0
        for i in range(image_embeds.shape[0]):
            img = image_embeds[i]
            sim = img @ text_feature.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)

        print("text 2 image")
        i = 0
        correct = 0
        total = 0
        for i in range(text_feature.shape[0]):
            text = text_feature[i]
            sim = text @ image_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)
