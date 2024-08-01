import os
import subprocess

# 定义参数
name = "eval_imagenet_1k"
eval_data_type = "ImageNet1K"
eval_test_data_dir = "/home/nyw/datasets/ImageNet1K/ImageNet1K/validation"
device_id = 0
checkpoint = "/home/nyw/contrastive_detection/rebuttal_nips/BadCLIP-master/epoch_10.pt"
add_backdoor = True
asr = True
patch_type = "ours_tnature"
patch_location = "middle"
label = "banana"
patch_folder = "/home/nyw/contrastive_detection/rebuttal_nips/BadCLIP-master/BadCLIP-master/opti_patches"
log_file = "/home/nyw/contrastive_detection/rebuttal_nips/BadCLIP-master/BadCLIP-master/experiment_log.txt"

# 获取文件夹中的所有图片文件
patch_files = [f for f in os.listdir(patch_folder) if os.path.isfile(os.path.join(patch_folder, f))]

# 遍历每个图片文件并运行命令
with open(log_file, "a") as log:
    for patch_name in patch_files:
        patch_path = os.path.join(patch_folder, patch_name)
        cmd = [
            "python", "-m", "src.main",
            "--name", name,
            "--eval_data_type", eval_data_type,
            "--eval_test_data_dir", eval_test_data_dir,
            "--device_id", str(device_id),
            "--checkpoint", checkpoint,
            "--add_backdoor",
            "--asr",
            "--patch_type", patch_type,
            "--patch_location", patch_location,
            "--label", label,
            "--patch_name", patch_path
        ]
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        # 记录输出到日志文件
        log.write(f"Patch: {patch_name}\n")
        log.write(result.stdout)
        log.write(result.stderr)
        log.write("\n" + "="*80 + "\n")

print("实验完成，结果已记录到日志中。")
