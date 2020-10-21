import psutil
import os
import time
import torch
from utils import get_example_inputs, gen_sentence
from transformers import BertModel, BertConfig

os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import onnxruntime
import numpy

if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    print("warning: onnxruntime-gpu is not built with OpenMP.")

sess_options = onnxruntime.SessionOptions()
sess_options.optimized_model_filepath = "onnx/optimized_model_cpu.onnx"

session = onnxruntime.InferenceSession("onnx/bert-base-cased.onnx",
                                       sess_options,
                                       providers=["CPUExecutionProvider"])

exam_text = ["best hotel in bay area", "here is an example of gpt2 model"]
exam_text = gen_sentence(20)
cache_dir = "./cache_models"
model_name_or_path = "bert-base-cased"
max_seq_length = 128
input_ids, attention_mask, position_ids, empty_past = get_example_inputs(prompt_text=exam_text,
                                                                         model_name_or_path=model_name_or_path,
                                                                         cache_dir=cache_dir)


config = BertConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
torch_model = BertModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
torch_model.eval().to(device)

latency = []
with torch.no_grad():
    for i, a, p in zip(input_ids, attention_mask, position_ids):
        inputs = {
            'input_ids': i.to(device).reshape(1, max_seq_length),
            'attention_mask': a.to(device).reshape(1, max_seq_length),
            'token_type_ids': p.to(device).reshape(1, max_seq_length)
        }
        start = time.time()
        outputs = torch_model(**inputs)
        latency.append(time.time() - start)
print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))

ort_inputs = {
    'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
    'attention_mask': numpy.ascontiguousarray(attention_mask.cpu().numpy()),
    'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy())
}
# for i, past_i in enumerate(empty_past):
#     ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())

latency = []
for i, a, p in zip(input_ids, attention_mask, position_ids):
    print(i)
    ort_inputs = {
        'input_ids':  i.cpu().reshape(1, max_seq_length).numpy(),
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