import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
# Test CUDA with a simple tensor operation
x = torch.randn(3, 3).cuda()
y = torch.randn(3, 3).cuda()
z = torch.matmul(x, y)
print("\nTest tensor multiplication on GPU:")
print(z) 