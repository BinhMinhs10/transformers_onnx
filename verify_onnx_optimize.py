import time
import os
import psutil
from utils import gen_sentence
from utils import get_example_inputs

import onnxruntime
import numpy

if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

    session = onnxruntime.InferenceSession("onnx/bert-base-cased.onnx",
                                           providers=["CPUExecutionProvider"])
    exam_text = ["best hotel in bay area", "here is an example of gpt2 model"]
    exam_text = gen_sentence(20)
    cache_dir = "./cache_models"
    model_name_or_path = "bert-base-cased"
    max_seq_length = 128
    input_ids, attention_mask, position_ids, empty_past = get_example_inputs(prompt_text=exam_text,
                                                                             model_name_or_path=model_name_or_path,
                                                                             cache_dir=cache_dir)
    latency = []
    for i, a, p in zip(input_ids, attention_mask, position_ids):
        ort_inputs = {
            'input_ids': i.cpu().reshape(1, max_seq_length).numpy(),
            'attention_mask': a.cpu().reshape(1, max_seq_length).numpy(),
            'token_type_ids': p.cpu().reshape(1, max_seq_length).numpy(),
        }
        start = time.time()
        ort_outputs_before = session.run(None, ort_inputs)
        latency.append(time.time() - start)
    print("OnnxRuntime cpu Inference time before = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

    session = onnxruntime.InferenceSession("onnx/optimized_model_cpu.onnx",
                                           providers=["CPUExecutionProvider"])

    cache_dir = "./cache_models"
    model_name_or_path = "bert-base-cased"
    max_seq_length = 128
    input_ids, attention_mask, position_ids, empty_past = get_example_inputs(prompt_text=exam_text,
                                                                             model_name_or_path=model_name_or_path,
                                                                             cache_dir=cache_dir)
    latency = []
    for i, a, p in zip(input_ids, attention_mask, position_ids):
        ort_inputs = {
            'input_ids': i.cpu().reshape(1, max_seq_length).numpy(),
            'attention_mask': a.cpu().reshape(1, max_seq_length).numpy(),
            'token_type_ids': p.cpu().reshape(1, max_seq_length).numpy(),
        }
        start = time.time()
        ort_outputs_after = session.run(None, ort_inputs)
        latency.append(time.time() - start)
    print("OnnxRuntime cpu Inference time after = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

    print("***** Verifying correctness *****")
    for i in range(2):
        print('PyTorch and ONNX Runtime output {} are close:'.format(i),
              numpy.allclose(ort_outputs_before[i], ort_outputs_after[i], rtol=1e-05, atol=1e-04))

