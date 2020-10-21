from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from transformers import AutoConfig
import torch
import os

# Create a cache directory to store pretrained model.
cache_dir = os.path.join(".", "cache_models")
onnx_model_path = "onnx/gpt2.onnx"
model_name_or_path = "gpt2"

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)

print(model.config)

num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer


Gpt2Helper.export_onnx(model, device, onnx_model_path)