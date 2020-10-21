from pathlib import Path
from transformers.convert_graph_to_onnx import convert
import shutil

# checking whether folder exists or not
# try:
#     shutil.rmtree("onnx")
# except OSError as e:
#     print("Error: %s" % e.strerror)


if __name__ == "__main__":
    name_model = "bert-base-cased"
    name_model = "gpt2"
    name_model = "vinai/phobert-base"
    convert(
        framework="pt",
        model=name_model,
        output=Path("onnx/" + name_model + ".onnx"),
        opset=11
    )