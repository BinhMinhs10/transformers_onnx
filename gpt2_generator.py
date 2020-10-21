from transformers import AutoConfig
from transformers import GPT2LMHeadModel
import torch
import os
import time
from utils import get_example_inputs, get_tokenizer, \
    inference_with_io_binding, test_generation
import onnxruntime
import numpy

exam_text = ['best hotel in bay area', 'here is an example of gpt2 model']
cache_dir = os.path.join(".", "cache_models")
onnx_model_path = "onnx/gpt2.onnx"
model_name_or_path = "gpt2"


config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
torch_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
torch_model.eval().to(device)

input_ids, attention_mask, position_ids, empty_past = get_example_inputs(prompt_text=exam_text)
print("input_ids", input_ids)
print("attention_mask", attention_mask)
print("position_ids", position_ids)

print(empty_past)
with torch.no_grad():
    torch_output = torch_model(
        input_ids,
        past=empty_past,
        attention_mask=attention_mask,
        position_ids=position_ids
    )

session = onnxruntime.InferenceSession(onnx_model_path)
ort_inputs = {
    'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
    'attention_mask': numpy.ascontiguousarray(attention_mask.cpu().numpy()),
    'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy())
}
for i, past_i in enumerate(empty_past):
    ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
ort_outputs = session.run(None, ort_inputs)

logits_masked_diff = (torch_output[0] - ort_outputs[0]) * attention_mask.unsqueeze(2)
max_logits_diff = logits_masked_diff.abs().max()
print("max logits diff (ignored padding", max_logits_diff)


outputs = inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, empty_past)
for i in range(len(outputs)):
    assert torch.eq(outputs[i], torch.from_numpy(ort_outputs[i])).all()
print("="*8 + "IO Binding result is good" + "="*8)


tokenizer = get_tokenizer(model_name_or_path, cache_dir)
input_text = exam_text


input_text = []
for i in range(10):
    input_text.append("best hotel in bay area")

start = time.time()
test_generation(torch_model, tokenizer, config=config, input_text=input_text, ort_session=session)
print(time.time() - start)
start = time.time()
test_generation(torch_model, tokenizer, config=config, input_text=input_text)
print(time.time() - start)
