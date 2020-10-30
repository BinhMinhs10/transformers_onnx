import os
import torch
import onnxruntime
from transformers import RobertaForMaskedLM, RobertaConfig, PhobertTokenizer
from phobert_model.phobert_utils import to_numpy
import numpy as np
device = torch.device("cpu")

output_dir = os.path.join("..", "onnx")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

export_model_path = os.path.join(output_dir, "phobert-base-formaskedlm.onnx")
model_name_or_path = "vinai/phobert-base"
cache_dir = "../cache_models"
enable_overwrite = True

tokenizer = PhobertTokenizer.from_pretrained(model_name_or_path)
input_ids = torch.tensor(tokenizer.encode(
    "Hôm nay trời <mask> quá",
    add_special_tokens=True)
).unsqueeze(0)


ort_session = onnxruntime.InferenceSession(export_model_path)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
ort_out = ort_session.run(None, ort_inputs)
print(len(ort_out[0][0][5]))

config = RobertaConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = RobertaForMaskedLM.from_pretrained(
    model_name_or_path,
    config=config,
    cache_dir=cache_dir
)
with torch.no_grad():
    out = model(input_ids)

print("***** Verifying correctness *****")
print('PyTorch and ONNX Runtime output are close: {}'.format(
    np.allclose(to_numpy(out[0]), ort_out[0], rtol=1e-03, atol=1e-04)))
