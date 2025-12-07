
from model import himo as longclip

import torch
from torchvision.datasets import CocoCaptions
from PIL import Image
import numpy as np
import time

data_root = "data/flickr/flickr30k-images/"

def get_text_feature(model, device, data_file):
    text_list = []
    feature_list = []
    with torch.no_grad():
        with open(data_file, 'r') as f:
            dataset = f.readlines()
            for data in dataset:
                image = data.split('\t')[0]
                text = data.split('\t')[1]
                text_list.append(text)
        len_list = len(text_list)
        print(len_list)
     
    #avoid OOM
    time_stamp = time.time()
    with torch.no_grad():
        split_num = 160 # 20 # 160
        bsz = len_list // split_num + 1
        for i in range(split_num):
            if i * bsz >= len_list: break
            text = text_list[i*bsz: (i+1)*bsz]
            if i % 20 == 0:
                cost = time.time() - time_stamp
                time_stamp = time.time()
                print(f"text feature: {i}/{split_num}, cost={cost:.4f}s.")

            text = longclip.tokenize(text, truncate=True).to(device)
            feature_list.append(model.encode_text(text).to('cpu'))
    
    text_feature = torch.concatenate(feature_list, dim=0)
    return text_feature
    

def get_image_feature(model, preprocess, device, data_file):
    img_feature_list = []
    time_stamp = time.time()
    
    with torch.no_grad():
        with open(data_file, 'r') as f:
            dataset = f.readlines()
            data_len = len(dataset)
            for i in range(data_len//5):
                if i % 50 == 0:
                    cost = time.time() - time_stamp
                    time_stamp = time.time()
                    print(f"extract img feat: {i}/{data_len//5}, cost={cost:.4f}s.")

                #1 image corresponding to 5 captions
                data = dataset[5*i]
                image_name = data.split('\t')[0][:-2]
                image = Image.open(data_root + image_name)
                image = preprocess(image).unsqueeze(0).to(device)
                img_feature = model.encode_image(image).to('cpu')
                img_feature_list.append(img_feature)
                torch.cuda.empty_cache()
                del img_feature, image

            img_feature = torch.concatenate(img_feature_list, dim=0)
            return img_feature

def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (text_feature @ image_feature.T).softmax(dim=-1)

        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1

        print(pred_true/text_feature.shape[0])

def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break

        print(pred_true/image_feature.shape[0])

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1]

    model, preprocess = longclip.load(model_path, device=device)
    model.eval()
    language = sys.argv[2]
    if language == "en":
        data_file = "data/flickr/test_1k.txt"

    text_feature = get_text_feature(model, device, data_file)
    image_feature = get_image_feature(model, preprocess, device, data_file)

    print(f"model: {model_path}")
    print(f"flickr {language} i2t")
    get_accuracy_i2t(text_feature, image_feature, 1)
    get_accuracy_i2t(text_feature, image_feature, 5)
    get_accuracy_i2t(text_feature, image_feature, 10)
    print(f"flickr {language} t2i")
    get_accuracy_t2i(text_feature, image_feature, 1)
    get_accuracy_t2i(text_feature, image_feature, 5)
    get_accuracy_t2i(text_feature, image_feature, 10)