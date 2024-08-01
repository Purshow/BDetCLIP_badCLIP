'''
python -m backdoor_isolation \
    --name def_clean \
    --train_data data/GCC-training/backdoor_banana_blended_blended_16_500000_1500_train.csv \
    --device_id 2 \
    --pretrained \
'''
import os
from tqdm import tqdm
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from src.train import get_loss
import pandas as pd
import torch
import logging
import warnings
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import clip
from pkgs.openai.clip import load as load_model
from src.parser import parse_args
from src.logger import get_logger
from PIL import Image

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def worker(rank,options,logger):
    device = options.device
    torch.cuda.set_device(2)
    model, preprocess  = load_model(name = options.model_name, pretrained = options.pretrained)
    model.to(options.device)
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint  = torch.load(options.checkpoint, map_location = options.device)
            if options.complete_finetune or 'epoch' not in checkpoint:
                start_epoch = 0 
            # start_epoch = 0 if options.complete_finetune else checkpoint['epoch'] 
            state_dict  = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}

            model.load_state_dict(state_dict)
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # 从 CSV 文件中读取数据
    csv_file = options.train_data  # 替换为你的 CSV 文件路径
    root = os.path.dirname(csv_file)
    data = pd.read_csv(csv_file)

    # 初始化一个空列表来存储余弦相似度和相应的数据
    data_with_gab_similarity = pd.DataFrame(columns=['image', 'top_cosine_similarity', 'top_label', 'target_cosine_similarity', 'target_label', 'gap_position', 'gap_cosine_similarity'])

    config = eval(open(f"data/ImageNet1K/validation/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    label_id = int([i for i, x in enumerate(classes) if x == options.label][0])

    with torch.no_grad():
        text_embeddings = []
        if options.asr:
            backdoor_target_index = list(filter(lambda x: 'banana' in classes[x], range(len(classes))))
            backdoor_target_index = torch.tensor(backdoor_target_index[0]).to(options.device)
        for c in tqdm(classes):
            if options.patch_type is not None:
                if ('vqa' in options.patch_type):
                    text = ['remember ' + template(c) for template in templates]
                else:
                    text = [template(c) for template in templates]
            else:
                text = [template(c) for template in templates]
            text_tokens = preprocess.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)

    # 循环处理每一行数据
    topk = [1, 2, 3, 4, 5]
    correct = {k: 0 for k in topk}
    total = 0

    count_num = 0
    for index, row in tqdm(data.iterrows(), total=len(data)):
        # 图片预处理
        image_path = row["image"]  # 假设你的 CSV 文件中有一个名为 "image" 的列存储图片路径
        image = preprocess.process_image(Image.open(os.path.join(root, image_path))).unsqueeze(0).to(device)

        # 文本编码
        # text_input = row["caption"]  # 假设你的 CSV 文件中有一个名为 "caption" 的列存储文本描述
        # # text = clip.tokenize([text_input]).to(device)
        # text = preprocess.process_text(text_input)
        # input_ids=text['input_ids'].to(options.device)
        # attention_mask=text['attention_mask'].to(options.device)
        # 获取 CLIP 模型的输出 
        with torch.no_grad():
            image_features = model.get_image_features(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算余弦相似度
            logits = (image_features @ text_embeddings)
            sorted_scores, sorted_indices = torch.sort(logits, descending=True, dim=1)
            top_label = sorted_indices[0,0].item()
            top_cosine_similarity = sorted_scores[0,0].item()
            matches = torch.eq(sorted_indices, label_id)
            target_ranking_positions = torch.nonzero(matches, as_tuple=True)[1].item()
            target_label = sorted_indices[0, target_ranking_positions].item()
            target_cosine_similarity = sorted_scores[0, target_ranking_positions].item()
            if target_label != label_id:
                print('-------------------------Error-----------------------------')
            gap_cosine_similarity = top_cosine_similarity - target_cosine_similarity
            gap_position = target_ranking_positions
            # ranks = logits.topk(max(topk), 1)[1].T
            # top_label=ranks[0, 0].item()
            # second_label = ranks[1, 0].item()
            # expanded_indices = ranks.expand(-1, logits.size(1))
            # selected_logits = torch.gather(logits.expand_as(expanded_indices), 1, expanded_indices)
            # final_values = selected_logits[:, 0]
            # top_cosine_similarity = logits[:, top_label].item()
            # second_cosine_similarity = logits[:, second_label].item()
            # gap = top_cosine_similarity - second_cosine_similarity
            # target_cosine_similarity = logits[:, label_id].item()

            # for item in enurmrate(ranks[1:, 0]):
            #     if item == label_id:
            #         data_with_second_similarity = data_with_second_similarity._append({'image':image_path, 'top_cosine_similarity':top_cosine_similarity, 'top_label':top_label, 'second_cosine_similarity':second_cosine_similarity, 'second_label:':second_label, "gap":gap},ignore_index=True)
            #         count_num = count_num+1
            #         print(count_num)

        # if index < 1500:
        #     continue
        data_with_gab_similarity=data_with_gab_similarity._append({'image':image_path, 'top_cosine_similarity':top_cosine_similarity, 'top_label':top_label, 'target_cosine_similarity':target_cosine_similarity, 'target_label':target_label, 'gap_position':gap_position, 'gap_cosine_similarity':gap_cosine_similarity},ignore_index=True)
        if index > 1500:
            break

    data_with_gab_similarity_sorted = data_with_gab_similarity.sort_values(
        by=['gap_position', 'gap_cosine_similarity'], 
        ascending=[True, True]
    )
    print(data_with_gab_similarity_sorted['top_cosine_similarity'].mean())

    # gab_similarity_csv_file = f"backdoor/gap_similarity_{options.label}.csv"  # 新 CSV 文件路径
    # data_with_gab_similarity_sorted.to_csv(gab_similarity_csv_file, index=False)
    # print('save files: ' + gab_similarity_csv_file)


if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()

