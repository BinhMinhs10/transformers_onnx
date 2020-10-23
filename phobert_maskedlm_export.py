import os
import torch
from transformers import RobertaForMaskedLM, RobertaConfig, PhobertTokenizer
device = torch.device("cpu")

model_name_or_path = "vinai/phobert-base"
cache_dir = "./cache_models"
enable_overwrite = True

tokenizer = PhobertTokenizer.from_pretrained(model_name_or_path)
input_ids = torch.tensor(tokenizer.encode(
    "Hôm nay trời <mask> quá",
    add_special_tokens=True)
).unsqueeze(0)

config = RobertaConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = RobertaForMaskedLM.from_pretrained(
    model_name_or_path,
    config=config,
    cache_dir=cache_dir
)

output_dir = os.path.join(".", "onnx")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir,
                                 "phobert-base-formaskedlm.onnx")

if enable_overwrite:
    with torch.no_grad():
        symbolic_names = {0: 'batch_size',
                          1: 'max_seq_len'}
        torch.onnx.export(model,
                          (input_ids),
                          f=export_model_path,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size',
                                                  1: 'sentence_length',
                                                  2: 'attention_mask'},
                                        'output': {0: 'batch_size',
                                                   1: 'sentence_length'}})

print("="*8 + "Export model PhobertForMaskedLM ONNX" + "="*8)
