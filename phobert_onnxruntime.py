import psutil
import os
import time
import torch
from phobert_utils import get_example_inputs
from transformers import RobertaForMaskedLM, RobertaConfig, RobertaModel
import onnxruntime
import numpy


os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'


if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    print("warning: onnxruntime-gpu is not built with OpenMP.")

session = onnxruntime.InferenceSession("onnx/vinai/phobert-base.onnx",
                                       providers=["CPUExecutionProvider"])

exam_text = ["Hôm nay trời đẹp quá", "Người đàn ông bị rắn hổ mang chúa cắn"]
# exam_text = gen_sentence(20)
cache_dir = "./cache_models"
model_name_or_path = "vinai/phobert-base"
max_seq_length = 128
input_ids, attention_mask, token_type_ids = get_example_inputs(prompt_text=exam_text,
                                                               model_name_or_path=model_name_or_path,
                                                               cache_dir=cache_dir)

config = RobertaConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
torch_model = RobertaModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
torch_model.eval().to(device)

latency = []
with torch.no_grad():
    for i, a, p in zip(input_ids, attention_mask, token_type_ids):
        inputs = {
            'input_ids': i.to(device).reshape(1, max_seq_length),
            'attention_mask': a.to(device).reshape(1, max_seq_length),
            'token_type_ids': p.to(device).reshape(1, max_seq_length)
        }
        start = time.time()
        outputs = torch_model(**inputs)
        latency.append(time.time() - start)
print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))


latency = []
for i, a, p in zip(input_ids, attention_mask, token_type_ids):
    ort_inputs = {
        'input_ids': i.cpu().reshape(1, max_seq_length).numpy(),
        'attention_mask': a.cpu().reshape(1, max_seq_length).numpy(),
        'token_type_ids': p.cpu().reshape(1, max_seq_length).numpy(),
    }
    start = time.time()
    ort_outputs = session.run(None, ort_inputs)
    latency.append(time.time() - start)
print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

print("***** Verifying correctness *****")
for i in range(2):
    print('PyTorch and ONNX Runtime output {} are close:'.format(i),
          numpy.allclose(ort_outputs[i], outputs[i].cpu(), rtol=1e-05, atol=1e-04))
