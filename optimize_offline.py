from onnxruntime_tools import optimizer

model_name_or_path = "onnx/bert-base-cased.onnx"
model_name_or_path = "onnx/vinai/phobert-base_chatbot.onnx"


optimized_model = optimizer.optimize_model(model_name_or_path,
                                           model_type='bert',
                                           num_heads=12,
                                           hidden_size=768)
optimized_model.save_model_to_file("onnx/optimized_model_cpu.onnx")
print("="*8 + "Offline fully optimize" + "="*8)

