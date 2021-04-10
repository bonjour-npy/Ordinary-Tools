import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.load('filename.pth').to(device)
model_name = 'u2netp'
model_dir = os.path.join(os.getcwd() + os.sep + "saved_models" + os.sep + model_name + os.sep + model_name + '.pth')
print(model_dir)
# model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
onnx_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
model = torch.load(model_dir, map_location=device)
model.eval()
batch_size = 1  # 批处理大小
input_shape = (3, 320, 320)   # 输入数据

input_data_shape = torch.randn(batch_size, *input_shape, device=device)

torch.onnx.export(model, input_data_shape, onnx_dir + ".onnx", verbose=True)

