# Implement transformers
| Framework / Tool | Source code | 
| --- | --- |
| pytorch | [pytorch](https://github.com/BinhMinhs10/transformers_onnx/tree/master/transformer_pytorch) |
| tensorflow | [tf](https://github.com/BinhMinhs10/transformers_onnx/tree/master/transformer_tf_translation) | 

# Inference ONNX Model with ONNX Runtime
[link refer](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb)
* speed up 2x
* For CPU, optimized graph is slightly different: FastGelu is replaced by BiasGelu.
* Note that ONNX Runtime is compatible with Python versions 3.5 to 3.7.

## What is ONNX Runtime? (vnese)
[link refer](https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release/)
* tackled optimizing một model cho các môi trường (cloud GPU, desktop CPU,..) tốn nhiều thời gian

## Export the loaded model
```bash
python convert_onnx.py
```
## Inference ONNX Model across multiple platforms
```bash
python bert_onnxruntime.py
```
## Offline optimization
* sometime OnnxRuntime cannot be fully optimized:
    * new subgraph generated by new export tool and not covered by older version of OnnxRuntime
    * exported model uses dynamic axis, make harder for shape inference
    * some optimization is better to done offline. Like change input tensor type from float32 to float16 avoid Cast nodes to achieve better performance in V100 and T4 GPU
```bash
python experiment.py
```
