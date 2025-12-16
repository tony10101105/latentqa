import torch
import time

# Preallocate to avoid allocation overhead
x = torch.randn(8192, 8192*2, device="cuda")

while True:
    try:
        # Multiple operations to keep GPU busy
        y = x @ x.T
        z = torch.matmul(y, x)
        w = torch.nn.functional.relu(z)
        _ = w.sum()
        torch.cuda.synchronize()
        time.sleep(0.5)
    except Exception as e:
        print("keepalive error:", ed)