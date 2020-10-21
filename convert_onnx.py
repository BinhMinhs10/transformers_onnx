from pathlib import Path
from transformers.convert_graph_to_onnx import convert
import shutil

# checking whether folder exists or not
try:
    shutil.rmtree("onnx")
except OSError as e:
    print("Error: %s" % e.strerror)


convert(framework="pt",
        model="bert-base-cased",
        output=Path("onnx/bert-base-cased.onnx"),
        opset=11)
