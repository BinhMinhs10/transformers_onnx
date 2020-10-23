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

exam_text = ["Hôm nay trời đẹp quá", "Người đàn ông bị rắn hổ mang chúa cắn",
             "sát hại cô gái trẻ rồi chôn vùi bên bờ suối ở văn_bàn lào cai"]
# exam_text = gen_sentence(20)
cache_dir = "./cache_models"
model_name_or_path = "vinai/phobert-base"
max_seq_length = 128
input_ids, attention_mask, token_type_ids = get_example_inputs(prompt_text=exam_text,
                                                               model_name_or_path=model_name_or_path,
                                                               cache_dir=cache_dir)

latency = []
for i, a, p in zip(input_ids, attention_mask, token_type_ids):
    ort_inputs = {
        'input_ids': i.cpu().reshape(1, max_seq_length).numpy(),
        'attention_mask': a.cpu().reshape(1, max_seq_length).numpy(),
        'token_type_ids': p.cpu().reshape(1, max_seq_length).numpy(),
    }

    start = time.time()
    ort_outputs = session.run(None, ort_inputs)
    print(ort_outputs[0].shape)
    print(ort_outputs[1].shape)
    latency.append(time.time() - start)
print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))
