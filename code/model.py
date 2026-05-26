import torch
import torch.nn as nn
from transformers import CLIPVisionModel , CLIPImageProcessor
from transformers import OPTForCausalLM, AutoTokenizer

try:
    from dataset import FlickrDataset
except ImportError:
    from .dataset import FlickrDataset

class CLIPVisionEncoder(nn.Module):
    def __init__(self , model_name="openai/clip-vit-base-patch32"):
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
    """
    Q-Former
    """
    def __init__(self, num_query_tokens=32, hidden_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        #创建32个空白问题
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_dim)
        )
        # 让query token 读取 image token
        self.layers = nn.ModuleList([
            MiniQFormerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        #设定projection layer 对齐向量
        self.llm_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, image_embeds):

        B = image_embeds.shape[0]

        query_tokens = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            query_tokens = layer(query_tokens, image_embeds)

        query_tokens = self.llm_proj(query_tokens)

        return query_tokens


class MiniOPT(nn.Module):
    """
    加载OPT模型并冻结
    """
    def __init__(self, model_name="facebook/opt-125m"):
        super().__init__()

        self.opt_model = OPTForCausalLM.from_pretrained(model_name)
        self.opt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token

        for p in self.opt_model.parameters():
            p.requires_grad = False

    def forward(self, qformer_output, caption):
        device = qformer_output.device

        tokens = self.opt_tokenizer(
            caption,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        ).to(device)

        input_ids = tokens.input_ids
        text_attention_mask  = tokens.attention_mask

        text_embeds = self.opt_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat(
            [qformer_output , text_embeds],
             dim = 1
        )

        qformer_attention_mask = torch.ones(
            qformer_output.size()[:-1],
            dtype=torch.long,
            device = device
        )

        attention_mask = torch.cat(
            [qformer_attention_mask, text_attention_mask],
            dim=1
        )

        text_labels = input_ids.clone()
        text_labels[text_attention_mask == 0] = -100

        qformer_labels = torch.full(
            qformer_output.size()[:-1],
            -100,
            dtype=torch.long,
            device=device
        )

        labels = torch.cat(
            [qformer_labels, text_labels],
            dim=1
        )

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    @torch.no_grad()
    def generate_caption(self, qformer_output, max_new_tokens=30, prompt="A photo of"):
        device = qformer_output.device
        batch_size = qformer_output.size(0)
        prompts = [prompt] * batch_size
        prompt_tokens = self.opt_tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(device)
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        inputs_embeds = torch.cat(
            [qformer_output, prompt_embeds],
            dim=1
        )

        qformer_attention_mask = torch.ones(
            qformer_output.size()[:-1],
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.cat(
            [qformer_attention_mask, prompt_tokens.attention_mask],
            dim=1
        )

        generated_ids = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=self.opt_tokenizer.eos_token_id,
            eos_token_id=self.opt_tokenizer.eos_token_id
        )

        captions = self.opt_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        return [caption.strip() for caption in captions]


class MiniBLIP2(nn.Module):
    """
    简化版 BLIP-2: frozen CLIP -> trainable Mini Q-Former -> frozen OPT.
    """
    def __init__(
        self,
        vision_model_name="openai/clip-vit-base-patch32",
        opt_model_name="facebook/opt-125m"
    ):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(vision_model_name)
        self.qformer = MiniQFormer()
        self.opt = MiniOPT(opt_model_name)

    def forward(self, images, captions):
        image_embeds = self.vision_encoder(images)
        image_embeds = image_embeds.to(next(self.qformer.parameters()).device)
        qformer_output = self.qformer(image_embeds)
        return self.opt(qformer_output, captions)

    @torch.no_grad()
    def generate_caption(self, images, max_new_tokens=20, prompt="A photo of"):
        image_embeds = self.vision_encoder(images)
        image_embeds = image_embeds.to(next(self.qformer.parameters()).device)
        qformer_output = self.qformer(image_embeds)
        return self.opt.generate_caption(
            qformer_output,
            max_new_tokens=max_new_tokens,
            prompt=prompt
        )



# 测试代码
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Root_dir = "../data/Image"
    Caption_dir = "../data/captions.txt"
    FlickrData = FlickrDataset(Root_dir,Caption_dir)

    model = MiniBLIP2().to(device)
    img , caption = FlickrData[0]
    outputs = model([img], [caption])
    print(outputs.logits.shape)
    print(outputs.loss.item())









