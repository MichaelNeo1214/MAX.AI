import torch


print("PyTorch version:")
print(torch.__version__)

print()

print("CPU tersedia:")
print(torch.device("cpu"))

print()

x = torch.tensor([1, 2, 3, 4])

print("Tensor:")
print(x)

print()

print("Tensor x 2:")
print(x * 2)