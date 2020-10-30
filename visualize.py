import netron

# Change it to False to skip viewing the optimized model in browser.
enable_netron = True
if enable_netron:
    netron.start("onnx/optimized_model_cpu.onnx")
    # netron.start("onnx/bert-base-cased.onnx")
    # netron.start("onnx/vinai/phobert-base.onnx")
    # netron.start("onnx/phobert-base-formaskedlm.onnx")
