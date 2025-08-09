import sys

import torch

print(f"Python version   : {sys.version}")
print(f"Torch version    : {torch.__version__}")
print(f"CUDA available?  : {torch.cuda.is_available()}")
print(f"CUDA version     : {torch.version.cuda}")
print(f"GPU count        : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name         : {torch.cuda.get_device_name(0)}")
    x = torch.rand(3, 3).to("cuda")
    print(f"Tensor device    : {x.device}")
else:
    print("⚠️  GPU(CUDA)를 사용할 수 없습니다. CPU로만 동작합니다.")
