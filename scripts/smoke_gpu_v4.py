"""V4 primitive hardware/runtime check, only in an allocated single-GPU job."""
import os
from importlib.metadata import version

import torch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Use Slurm compute; never the login node")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
expected = os.environ.get("PRISM_GPU_MODEL", "A100").split("|")
assert all(expected) and any(x in torch.cuda.get_device_name(0) for x in expected)
assert torch.cuda.get_device_capability(0) >= (8, 0)
assert torch.__version__.split("+")[0] == "2.7.1" and torch.version.cuda == "12.6"
for name, value in {"transformers": "4.57.1", "tokenizers": "0.22.1", "huggingface-hub": "0.36.0",
                    "peft": "0.15.2", "accelerate": "1.7.0", "scipy": "1.15.3"}.items():
    assert version(name) == value, (name, version(name))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
x = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print("V4 GPU smoke passed:", torch.cuda.get_device_name(0))
