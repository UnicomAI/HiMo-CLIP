import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from model import himo as longclip


def pearson_correlation(x, y):
    """计算 Pearson 相关系数，x/y 为一维数组。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x 和 y 的长度必须相同")

    x_mean = x.mean()
    y_mean = y.mean()

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))

    if den == 0:
        return 0.0
    return num / den


def load_jsonl(file_path: str):
    """读取 jsonl 文件，返回一个 list[dict]。"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_himo_k(model, preprocess, data, device, image_root, print_freq=100):
    """
    直接对每个 case 做推理，只计算：
      - HiMo@K 的 Pearson（caption 索引 vs 相似度的相关系数），最后取平均
    不再计算任何 pairwise accuracy。
    """
    pearson_scores = []   # 每张图的 Pearson

    model.eval()

    for idx, ele in enumerate(data):
        img_path = os.path.join(image_root, ele["img_path"])
        caption_list = ele["caption"]


        # 加载图片
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] fail to open image {img_path}: {e}")
            continue

        # 预处理 + 编码
        image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embeds = model.encode_image(image)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # 一次性编码所有 caption
            text_tokens = longclip.tokenize(caption_list, truncate=True).to(device)
            text_embeds = model.encode_text(text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            sims = (image_embeds @ text_embeds.T).squeeze(0)  # [num_captions]
            sims_np = sims.detach().cpu().numpy().astype(float)

        # HiMo@K Pearson：索引 1..K 与相似度之间的相关系数
        x = np.arange(1, len(caption_list) + 1, dtype=float)
        p = pearson_correlation(x, sims_np)
        pearson_scores.append(p)

        if (idx + 1) % print_freq == 0:
            print(f"{idx + 1}/{len(data)} images processed")

    avg_pearson = float(np.mean(pearson_scores)) if pearson_scores else 0.0

    print(f"HiMo@K average Pearson: {avg_pearson:.4f}")

    return avg_pearson


def run(model_path, jobname, output_root, image_root):
    output_dir = os.path.join(output_root, jobname)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Load model from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(model_path, device=device)
    print("Model loaded.")

    data_file = "./himo_docci_data.json"
    task_data = load_jsonl(data_file)
    print(f"HiMo-Docci num: {len(task_data)}")

    print("===> Evaluating HiMo@K (Pearson only) ...")
    avg_pearson = evaluate_himo_k(model, preprocess, task_data, device, image_root)

    print(f"{model_path} | HiMo@K Pearson = {avg_pearson:.4f}")

    # 保存结果，方便多模型对比
    metrics = {
        "model": model_path,
        "HiMoK_pearson": avg_pearson,
    }
    out_path = os.path.join(output_dir, "himo_k_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    output_root = "./"
    image_root = "data/docci/images/"

    model_path = sys.argv[1]
    jobname = sys.argv[2]

    run(model_path, jobname, output_root, image_root)
