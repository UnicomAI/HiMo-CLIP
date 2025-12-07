import json
import cv2
from PIL import Image
from model import himo as longclip


import torch
import torch.utils.data as data
import os
import numpy as np
import json

image_root = 'data/docci/images'
caption_root_dict = {
    "en": 'data/docci/test_set.jsonl'
}
class local_dataset(data.Dataset):
    def __init__(self, image_root, caption_file):
        self.image_root = image_root
        data = []
        with open(caption_file, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info = self.data[index]
        caption = info["description"].strip()
        img_name = info["image_file"]
        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path)
        return image, caption

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1]
    model, preprocess = longclip.load(model_path, device=device)
    model.eval()
    print('model done!')

    language = sys.argv[2]
    caption_root = caption_root_dict[language]

    dataset = local_dataset(image_root=image_root, caption_file=caption_root)
    print(f"docci, lang: {language}, model: {model_path}")
    img_feature_list = []
    text_feature_list = []

    text_list = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            _, caption = dataset[i]
            text_list.append(caption)

        len_list = len(text_list)
        split_num = 5 # 20 # 160
        bsz = len_list // split_num + 1
        for i in range(split_num):
            if i * bsz >= len_list: break
            text = text_list[i*bsz: (i+1)*bsz]
            text_input = longclip.tokenize(text, truncate=True).to(device)
            text_feature_list.append(model.encode_text(text_input))
    
        text_feature = torch.concatenate(text_feature_list, dim=0)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        print(f"text_feature.shape: {text_feature.shape}")
        
        for i, (image, caption) in enumerate(dataset):            
            image_inputs = preprocess(image).unsqueeze(0).to(device)
            img_feature = model.encode_image(image_inputs)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.stack(img_feature_list).squeeze()
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        print(f"image_feature.shape: {image_embeds.shape}")

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