import os
import csv
import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
import pandas as pd
from src.data import ImageLabelDataset#gai
import torch.nn.functional as F
from .scheduler import cosine_scheduler
import json
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ROC

def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking = True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics

def count_files_in_directory(self, directory_path):
    all_items = os.listdir(directory_path)
    
    files = [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]
    
    return len(files)
"""
def get_zeroshot_metrics(model, processor, test_dataloader, options):
    logging.info("Started zeroshot testing")

    model.eval()
    umodel = model.module if(options.distributed) else model
    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]

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
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)
        
    with torch.no_grad():
        topk = [1, 3, 5, 10]
        if not(options.eval_test_data_csv is None):
            labeled_bool = []
        correct = {k: 0 for k in topk}
        total = 0
        for image, label in tqdm(test_dataloader):
            image, label = image.to(options.device), label.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == 954
            if not(options.eval_test_data_csv is None):
                transposed_predictions = predictions.t()
                for t in transposed_predictions:
                    labeled_bool.append(t[0])
            total += predictions.shape[1]
            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

# 假设你已经有了一个打开的文件句柄，命名为result_file
# 你可以在循环外部打开这个文件，并在循环结束后关闭它
        result_file = open('predictions_results.txt', 'w')

        for image, label in tqdm(test_dataloader):
            image, label = image.to(options.device), label.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == 954

            # 将预测结果转换为适合记录的格式
            predictions_list = ranks.tolist()

            # 将预测结果写入文件
            result_file.write(' '.join(map(str, predictions_list)) + '\n')

            # 如果需要，也可以在这里处理其他逻辑，比如计算top-k准确率等

            if not(options.eval_test_data_csv is None):
                transposed_predictions = predictions.t()
                for t in transposed_predictions:
                    labeled_bool.append(t[0])

            total += predictions.shape[1]
            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

        # 循环结束后，不要忘记关闭文件
        result_file.close()
    results = {f"zeroshot_top{k}": correct[k] / total for k in topk}
    if not(options.eval_test_data_csv is None):
        labeled_bool = torch.stack(labeled_bool, dim=0)
        df   = pd.read_csv(options.eval_test_data_csv, sep = ',')
        df['backdoor_lables'] = labeled_bool.cuda().cpu().numpy()
        df.to_csv(options.eval_test_data_csv.replace('is_backdoor', 'labeled_backdoor'))
    with open('results.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([options.name, str(results)])
    logging.info("Finished zeroshot testing")

    return results
"""

def get_zeroshot_metrics(model, processor, test_dataloader, options):

    logging.info("Started zeroshot testing")

    model.eval()
    umodel = model.module if(options.distributed) else model
    config = eval(open(f"/home/nyw/contrastive_detection/data/ImageNet1K/validation/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]

    json_file_path = '/home/nyw/contrastive_detection/related_class_sentences.json'

    with open(json_file_path, 'r') as json_file:
        class_data2 = json.load(json_file)

    un_class_data = {}

    with open('/home/nyw/contrastive_detection/unrelated_class_sentences.txt') as f:
        all_sentences = f.readlines()
        for i,sentence in enumerate(all_sentences):
            temp = sentence.split(' ', 1)[1]
            un_class_data[classes[i]] = temp.strip()
            un_class_data[classes[i]] =sentence
    
    with torch.no_grad():
        text_embeddings1 = []
        text_embeddings2 = []
        if options.asr:
            backdoor_target_index = list(filter(lambda x: "banana" in classes[x], range(len(classes)))) 
            backdoor_target_index = torch.tensor(backdoor_target_index[0]).to(options.device)
        for c in tqdm(classes):
            text11 = class_data2[c] 
            
            for i in range(len(text11)):
                temp = text11[i].split('. ', 1)[0]
                text11[i] = text11[i].replace(temp+'. ', '')
            
            text1 = text11
            text2 = [template(c).replace('.', ',')+' '+un_class_data[c].lower().replace('"', '') for template in templates]

            text_tokens1 = processor.process_text(text1)
            text_input_ids1, text_attention_mask1 = text_tokens1["input_ids"].to(options.device), text_tokens1["attention_mask"].to(options.device)
            text_embedding1 = umodel.get_text_features(input_ids = text_input_ids1, attention_mask = text_attention_mask1)
            text_embedding1 /= text_embedding1.norm(dim = -1, keepdim = True)
            text_embedding1 = text_embedding1.mean(dim = 0)
            text_embedding1 /= text_embedding1.norm()
            text_embeddings1.append(text_embedding1)

            text_tokens2 = processor.process_text(text2)
            text_input_ids2, text_attention_mask2 = text_tokens2["input_ids"].to(options.device), text_tokens2["attention_mask"].to(options.device)
            text_embedding2 = umodel.get_text_features(input_ids = text_input_ids2, attention_mask = text_attention_mask2)
            text_embedding2 /= text_embedding2.norm(dim = -1, keepdim = True)
            text_embedding2 = text_embedding2.mean(dim = 0)
            text_embedding2 /= text_embedding2.norm()
            text_embeddings2.append(text_embedding2)

        text_embeddings1 = torch.stack(text_embeddings1, dim=1).to(options.device)
        text_embeddings2 = torch.stack(text_embeddings2, dim=1).to(options.device)


    with torch.no_grad():
        all_diff = torch.zeros(50000)
        clean_indicator = torch.zeros(50000, dtype=int)
        bd_indicator = torch.zeros(50000, dtype=int)
        topk = [1, 3, 5, 10]
        correct1 = {k: 0 for k in topk}
        correct2 = {k: 0 for k in topk}
        total1 = 0
        total2 = 0
        results = []
        
        path_to_file_A = '/home/nyw/contrastive_detection/pic/bd1.txt'
        path_to_file_B = '/home/nyw/contrastive_detection/pic/bd0.txt'
        path_to_logits1 = '/home/nyw/contrastive_detection/2.txt'
        path_to_logits2 = '/home/nyw/contrastive_detection/1.txt'


        with open(path_to_file_A, 'w') as file_A, open(path_to_file_B, 'w') as file_B, \
            open(path_to_logits1, 'w') as logits1_file, open(path_to_logits2, 'w') as logits2_file:
            for image, label, is_bd, idx in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                image_embedding = umodel.get_image_features(image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                logits1 = (image_embedding @ text_embeddings1)
                logits2 = (image_embedding @ text_embeddings2)


                logits1_val = logits1.detach().clone().cpu()
                logits2_val = logits2.detach().clone().cpu()

                for i in range(len(logits1_val)):
                    logits1_file.write(f'{logits1_val[i].tolist()}\n')
                for i in range(len(logits2_val)):
                    logits2_file.write(f'{logits2_val[i].tolist()}\n')
                exp_logits1 = torch.exp(logits1)
                exp_logits2 = torch.exp(logits2)

                diff = ((logits1 - logits2) ).sum(dim=1)
                diff_values = diff.detach().clone().cpu()

                for i in range(len(is_bd)):
                    if is_bd[i] == 1:
                        file_A.write(f'{diff_values[i].item()}\n')
                    else:
                        file_B.write(f'{diff_values[i].item()}\n')

                all_diff[idx] = diff_values
                bd_indicator[idx] = is_bd.detach().clone()
                clean_indicator[idx] = 1 - bd_indicator[idx]

                ranks1 = logits1.topk(max(topk), 1)[1].T
                predictions1 = ranks1 == label
                total1 += predictions1.shape[1]
                for k in topk:
                    correct1[k] += torch.sum(torch.any(predictions1[:k], dim=0)).item()

                ranks2 = logits2.topk(max(topk), 1)[1].T
                predictions2 = ranks2 == label
                total2 += predictions2.shape[1]
                for k in topk:
                    correct2[k] += torch.sum(torch.any(predictions2[:k], dim=0)).item()

        selection = (all_diff < options.threshold).float()
        eval_selection(selection, bd_indicator, all_diff)



    results1 = {f"zeroshot_top{k}": correct1[k] / total1 for k in topk}
    with open('results.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([options.name, str(results1)])
    logging.info("Finished zeroshot testing")
    results2 = {f"zeroshot_top{k}": correct2[k] / total2 for k in topk}
    with open('results.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([options.name, str(results2)])
    logging.info("Finished zeroshot testing")

    return results



def eval_selection(pred,true,pos):

    accuracy = Accuracy(task='binary', num_classes=2)
    precision = Precision(task='binary', num_classes=2, average='micro')
    recall = Recall(task='binary', num_classes=2, average='micro')
    f1 = F1Score(task='binary', num_classes=2)
    auroc = AUROC(task='binary', num_classes=2)

    acc = accuracy(pred, true)
    pre = precision(pred, true)
    re = recall(pred, true)
    f1 = f1(pred, true)
    ar = 1-auroc(pos, true)

    print('accuracy:{}, precision:{}, recall:{}, f1:{}, auroc:{}.'.format(acc, pre, re, f1, ar))
class Finetune(torch.nn.Module):
    def __init__(self, input_dim, output_dim, model):
        super(Finetune, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.model  = model
    def forward(self, x):
        outputs = self.linear(self.model.get_image_features(x))
        return outputs

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def get_odim_metric(options):

    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    return output_dim, metric

def get_finetune_metrics(model, train_dataloader, test_dataloader, options):

    logging.info("Starting finetune testing")
    model.train()
    umodel = model.module if(options.distributed) else model

    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = Finetune(input_dim = input_dim, output_dim = output_dim, model = umodel).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))

    if options.checkpoint_finetune is not None:
        if(os.path.isfile(options.checkpoint_finetune)):
            checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
            if(not options.distributed and next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
            if(options.distributed and not next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {f'module.{key}': value for key, value in checkpoint.items()}
            state_dict = checkpoint["state_dict"]
            classifier.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint {options.checkpoint_finetune}")
    
    if(not options.checkpoint_finetune or not os.path.isfile(options.checkpoint_finetune)):
        for epoch in pbar:
            cbar = tqdm(train_dataloader, leave = False)
            for index, (image, label) in enumerate(cbar):
                step = len(train_dataloader) * epoch + index
                scheduler(step)
                image, label = image.to(options.device), label.to(options.device)
                logit = classifier(image)
                optimizer.zero_grad()
                loss = criterion(logit, label)
                loss.backward()
                optimizer.step()
                if options.wandb:
                    wandb.log({'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
                cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            if options.eval_frequency is not None:
                if (epoch % options.eval_frequency) == 0:
                    classifier.eval()
                    with torch.no_grad():
                        if(metric == "accuracy"):
                            correct = 0
                            for image, label in tqdm(test_dataloader):
                                image, label = image.to(options.device), label.to(options.device)
                                logits = classifier(image)
                                prediction = torch.argmax(logits, dim = 1)
                                if options.asr:
                                    non_label_indices = (label != 954).nonzero().squeeze()
                                    if type(non_label_indices) == int or len(non_label_indices):
                                        prediction = prediction[non_label_indices]
                                    correct += torch.sum(prediction == 954).item()
                                else:
                                    correct += torch.sum(prediction == label).item()
                    logging.info(f"EPOCH: {epoch}")
                    logging.info(f"linear_probe_accuracy: {correct / test_dataloader.num_samples}")
                    classifier.train()
            if not options.save_final:
                checkpoint = {'state_dict': classifier.state_dict()}
                checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
                os.makedirs(checkpoints_dir_path, exist_ok = True)
                pt_name = "finetune_" + str(epoch) + ".pt"
                torch.save(checkpoint, os.path.join(checkpoints_dir_path, pt_name))
        checkpoint = {'state_dict': classifier.state_dict()}
        checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(checkpoints_dir_path, exist_ok = True)
        torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"finetune.pt"))


    classifier.eval()
    
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(image)
                prediction = torch.argmax(logits, dim = 1)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
            logging.info(results)
            
    logging.info("Finished finetune testing")
    return results


def get_linear_probe_metrics(model, train_dataloader, test_dataloader, options):
    logging.info("Started linear probe testing")
    logging.info(f"Number of train examples: {train_dataloader.num_samples}")
    logging.info(f"Number of test examples: {test_dataloader.num_samples}")

    model.eval()
    umodel = model.module if(options.distributed) else model
    
    images = None
    labels = None
    with torch.no_grad():
        for image, label in tqdm(train_dataloader):
            image = umodel.get_image_features(image.to(options.device)).cpu()
            images = torch.cat([images, image], dim = 0) if(images is not None) else image
            labels = torch.cat([labels, label], dim = 0) if(labels is not None) else label

    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = options.batch_size, shuffle = True)
    
    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, options.lr, 0, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for index, (image, label) in enumerate(cbar):
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            image, label = image.to(options.device), label.to(options.device)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                prediction = torch.argmax(logits, dim = 1)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
        else:
            correct = torch.zeros(output_dim).to(options.device)
            total = torch.zeros(output_dim).to(options.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                predictions = torch.argmax(logits, dim = 1)
                
                temp = torch.zeros(output_dim, len(label)).to(options.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
        
    logging.info("Finished linear probe testing")
    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["eval_test"] is not None): 
            if(data["eval_train"] is not None):
                if options.linear_probe:
                    metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
                elif options.finetune:
                    metrics.update(get_finetune_metrics(model, data["eval_train"], data["eval_test"], options))
            else:
                metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options))
        
        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics