import torch
import torch.nn as nn
from transformers import CLIPVisionModel , CLIPImageProcessor
from dataset import FlickrDataset

class CLIPVisionEncoder(nn.Module):
    def __init__(self , model_name):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imageprocessor = CLIPImageProcessor.from_pretrained(model_name) #CLIP图像预处理
        self.vision_model = CLIPVisionModel.from_pretrained(model_name) #CLIP图像编码器
        self.vision_model.to(self.device) # 将编码器载入cuda
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def forward(self , inputs):
        pixel_values = self.imageprocessor(images=inputs, return_tensors="pt")
        pixel_values = pixel_values["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state

# Root_dir = "../data/Image"
# Caption_dir = "../data/captions.txt"
# FlickrData = FlickrDataset(Root_dir,Caption_dir)
#
# encoder = CLIPVisionEncoder("openai/clip-vit-base-patch32")
# img , caption = FlickrData[0]
# output = encoder(img)
# print(output.shape)





