import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FlickrDataset
from model import MiniBLIP2


def generate(num_samples=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(project_dir, "data", "Image")
    caption_dir = os.path.join(project_dir, "data", "captions.txt")
    checkpoint_path = os.path.join(
        project_dir,
        "outputs",
        "checkpoints",
        "mini_blip2_qformer.pt"
    )
    sample_dir = os.path.join(project_dir, "outputs", "samples")
    os.makedirs(sample_dir, exist_ok=True)

    dataset = FlickrDataset(root_dir, caption_dir)
    model = MiniBLIP2().to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.qformer.load_state_dict(checkpoint["qformer"])

    model.eval()
    rows = [
        "| 图片编号 | 真实 Caption | 模型生成 Caption |",
        "|---|---|---|"
    ]

    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            image, true_caption = dataset[idx]
            generated_caption = model.generate_caption([image])[0].strip()
            image_name = dataset.img_path[idx]
            rows.append(
                f"| {image_name} | {true_caption} | {generated_caption} |"
            )
            print(image_name)
            print("true:", true_caption)
            print("generated:", generated_caption)
            print()

    output_path = os.path.join(sample_dir, "generated_captions.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"saved generated captions to {output_path}")


if __name__ == "__main__":
    generate()
