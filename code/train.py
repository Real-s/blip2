import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FlickrDataset
from model import MiniBLIP2


def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)


def train(epochs=1, batch_size=1, learning_rate=1e-4, max_steps=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(project_dir, "data", "Image")
    caption_dir = os.path.join(project_dir, "data", "captions.txt")
    output_dir = os.path.join(project_dir, "outputs")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = FlickrDataset(root_dir, caption_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = MiniBLIP2().to(device)
    model.train()
    model.vision_encoder.eval()
    model.opt.eval()

    optimizer = torch.optim.AdamW(
        model.qformer.parameters(),
        lr=learning_rate
    )

    global_step = 0
    log_path = os.path.join(log_dir, "train_log.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(epochs):
            total_loss = 0.0
            epoch_steps = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for images, captions in progress_bar:
                optimizer.zero_grad()
                outputs = model(images, captions)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_steps += 1
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                log_file.write(
                    f"epoch={epoch + 1}, step={global_step}, loss={loss.item():.6f}\n"
                )
                log_file.flush()

                if max_steps is not None and global_step >= max_steps:
                    break

            avg_loss = total_loss / max(epoch_steps, 1)
            log_file.write(f"epoch={epoch + 1}, avg_loss={avg_loss:.6f}\n")

            if max_steps is not None and global_step >= max_steps:
                break

    checkpoint_path = os.path.join(checkpoint_dir, "mini_blip2_qformer.pt")
    torch.save(
        {
            "qformer": model.qformer.state_dict(),
        },
        checkpoint_path
    )
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved training log to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps
    )
