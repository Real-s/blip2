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


class MiniQFormerLayer(nn.Module):
    def __init__(self , hidden_dim = 768 , num_heads = 8):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, queries, image_embeds):
        # queries: [B,32,768]
        # image_embeds: [B,50,768]

        residual = queries
        x, _ = self.self_attn(
            queries,
            queries,
            queries
        )
        queries = self.norm1(residual + x)

        residual = queries
        x, _ = self.cross_attn(
            queries,
            image_embeds,
            image_embeds
        )
        queries = self.norm2(residual + x)

        residual = queries
        x = self.ffn(queries)
        queries = self.norm3(residual + x)

        return queries


class MiniQFormer(nn.Module):

    def __init__(self):
        super().__init__()
        #创建32个空白问题
        self.query_tokens = nn.Parameter(
            torch.randn(1, 32, 768)
        )
        # 让query token 读取 image token
        self.layers = nn.ModuleList([
            MiniQFormerLayer(768, 8),
            MiniQFormerLayer(768, 8)
        ])
        #设定projection layer 对齐向量
        self.llm_proj = nn.Linear(768, 768)

    def forward(self, image_embeds):

        B = image_embeds.shape[0]

        query_tokens = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            query_tokens = layer(query_tokens, image_embeds)

        query_tokens = self.llm_proj(query_tokens)

        return query_tokens



# 测试代码
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Root_dir = "../data/Image"
    Caption_dir = "../data/captions.txt"
    FlickrData = FlickrDataset(Root_dir,Caption_dir)

    encoder = CLIPVisionEncoder("openai/clip-vit-base-patch32")
    img , caption = FlickrData[0]
    image_embeds  = encoder(img)

    trans_qformer = MiniQFormer().to(device)
    q_out  = trans_qformer(image_embeds)

    print(trans_qformer.llm_proj)

    for name, p in trans_qformer.named_parameters():
        if "llm_proj" in name:
            print(name, p.requires_grad)

    # optimizer = torch.optim.AdamW(
    #     trans_qformer.parameters(),
    #     lr=1e-4
    # )
    # loss = q_out.mean()
    #
    # for step in range(20):
    #     image_embeds = encoder(img)
    #
    #     q_out = trans_qformer(image_embeds)
    #
    #     loss = q_out.mean()
    #
    #     optimizer.zero_grad()
    #
    #     loss.backward()
    #
    #     optimizer.step()
    #
    #     print(f"step:{step} loss:{loss.item()}")




    # for name, p in trans_qformer.named_parameters():
    #     print(name, p.requires_grad)
    # print(image_embeds.shape)
    # print(q_out.shape)
    # output = encoder(img)
    # print(output.shape)





