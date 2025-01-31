import os
import sys
current_directory = os.getcwd()
sys.path.insert(1,current_directory)
import torch
import pickle
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


def get_model(args, checkpoint):
    model, processor = load_model(name=args.model_name, pretrained=False)
    if (args.device == "cpu"): model.float()
    model.to(args.device)
    state_dict = torch.load(checkpoint, map_location=args.device)["state_dict"]
    if (next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, processor


class ImageCaptionDataset(Dataset):
    def __init__(self, path, images, captions, processor):
        self.root = os.path.dirname(path)
        self.processor = processor
        self.images = images
        self.captions = self.processor.process_text(captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
        return item


def get_embeddings(model, dataloader, processor, args):
    device = args.device
    list_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking=True), batch[
                "attention_mask"].to(device, non_blocking=True), batch["pixel_values"].to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            list_embeddings.append(outputs.image_embeds)
    return torch.cat(list_embeddings, dim=0).cpu().detach().numpy()


def plot_embeddings(args):
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_data):
        checkpoint = f'epoch_{args.epoch}.pt'
        if args.pretrained:
            model, processor = load_model(args.model_name, args.pretrained)
            if (args.device == "cpu"): model.float()
            model.to(args.device)
        else:
            model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
        df = pd.read_csv(args.original_csv)

        # we divide data into three categories -- clean, backdoored and detected, backdoored and undetected
        images, captions, is_backdoor, backdoor_lables = df['image'].tolist(), df['caption'].tolist(), df['is_backdoor'].tolist(), df['backdoor_lables'].tolist()
        backdoor_indices = list(filter(lambda x: 'backdoor' in images[x], range(len(images))))
        backdoor_detected_indices = [x for x in backdoor_indices if (is_backdoor and backdoor_lables[x]) is True]
        backdoor_undetected_indices = [x for x in backdoor_indices if (is_backdoor and backdoor_lables[x]) is False]

        clean_indices = list(filter(lambda x: 'backdoor' not in images[x], range(len(images))))
        clean_indices = clean_indices[:10000]


        backdoor_detected_images, backdoor_detected_captions = [images[x] for x in backdoor_detected_indices], \
                                                               [captions[x] for x in backdoor_detected_indices]

        backdoor_undetected_images, backdoor_undetected_captions = [images[x] for x in backdoor_undetected_indices], \
                                                               [captions[x] for x in backdoor_undetected_indices]

        clean_images, clean_captions = [images[x] for x in clean_indices], [captions[x] for x in clean_indices]

        dataset_clean = ImageCaptionDataset(args.original_csv, clean_images, clean_captions, processor)
        dataset_backdoor_detected = ImageCaptionDataset(args.original_csv, backdoor_detected_images,
                                                        backdoor_detected_captions, processor)
        dataset_backdoor_undetected = ImageCaptionDataset(args.original_csv, backdoor_undetected_images,
                                                        backdoor_undetected_captions, processor)

        dataloader_clean = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                         drop_last=False)
        dataloader_backdoor_detected = DataLoader(dataset_backdoor_detected, batch_size=args.batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=False)
        dataloader_backdoor_undetected = DataLoader(dataset_backdoor_undetected, batch_size=args.batch_size,
                                                    shuffle=False, pin_memory=True, drop_last=False)


        clean_images_embeddings = get_embeddings(model, dataloader_clean, processor, args)
        backdoor_detected_images_embeddings = get_embeddings(model, dataloader_backdoor_detected, processor, args)
        if len(dataloader_backdoor_undetected) > 0:
            backdoor_undetected_images_embeddings = get_embeddings(model, dataloader_backdoor_undetected, processor, args)

        len_clean = len(clean_images_embeddings)
        len_backdoor_detected = len(backdoor_detected_images_embeddings)
        if len(dataloader_backdoor_undetected) > 0:
            len_backdoor_undetected = len(backdoor_undetected_images_embeddings)
        else:
            len_backdoor_undetected = 0


        if len(dataloader_backdoor_undetected) > 0:
            all_embeddings = np.concatenate([clean_images_embeddings, backdoor_detected_images_embeddings, backdoor_undetected_images_embeddings], axis=0)
        else:
            all_embeddings = np.concatenate([clean_images_embeddings, backdoor_detected_images_embeddings], axis=0)
        print(len_clean)
        
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_embeddings, len_clean, len_backdoor_detected, len_backdoor_undetected), f)

    with open(args.save_data, 'rb') as f:
        all_embeddings, len_clean, len_backdoor_detected, len_backdoor_undetected = pickle.load(f)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)

    # pca = PCA(n_components = 2)
    # results = pca.fit_transform(all_embeddings)
    # print(pca.explained_variance_ratio_)

    plt.scatter(results[:len_clean, 0], results[:len_clean, 1], label='Clean')
    plt.scatter(results[len_clean:len_clean+len_backdoor_detected, 0], results[len_clean:len_clean+len_backdoor_detected, 1], label='Backdoor Detected')
    plt.scatter(results[len_clean + len_backdoor_detected:, 0],
                results[len_clean + len_backdoor_detected:, 1], label='Backdoor Undetected')

    plt.grid()
    plt.legend()
    plt.title(args.title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok=True)
    plt.savefig(args.save_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type=str, default=None, help="original csv with captions and images")
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    parser.add_argument("--model_name", type=str, default="RN50", choices=["RN50", "RN101", "RN50x4", "ViT-B/32"],
                        help="Model Name")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/clip/",
                        help="Path to checkpoint directories")
    parser.add_argument("--save_data", type=str, default=None, help="Save data")
    parser.add_argument("--save_fig", type=str, default=None, help="Save fig png")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument("--epoch", type=int, default=64, help="Epoch")
    parser.add_argument("--title", type=str, default=None, help="Title for plot")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

    args = parser.parse_args()

    plot_embeddings(args)