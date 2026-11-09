"""Run only inside an allocated single-GPU Yancheng Slurm job."""
import os
from importlib.metadata import version

import torch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run this inside a Slurm allocation, never on the login node")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert os.environ.get("PRISM_GPU_MODEL", "A100") in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) >= (8, 0)
assert version("torch").split("+")[0] == "2.7.1" and torch.version.cuda == "12.6"
for name, expected in {"transformers": "4.52.3", "trl": "0.15.2", "peft": "0.15.2", "accelerate": "1.7.0", "scipy": "1.15.3"}.items():
    assert version(name) == expected, (name, version(name))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print("GPU smoke passed:", torch.cuda.get_device_name(0))
