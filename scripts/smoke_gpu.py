"""Run only inside an allocated Yancheng Slurm GPU job."""
import os
from importlib.metadata import version

import torch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run this inside a Slurm allocation, never on the login node")
expected_count = int(os.environ.get("PRISM_EXPECTED_GPU_COUNT", "1"))
assert torch.cuda.is_available() and torch.cuda.device_count() == expected_count
expected_models = os.environ.get("PRISM_GPU_MODEL", "A100").split("|")
names = [torch.cuda.get_device_name(i) for i in range(expected_count)]
assert all(expected_models) and all(any(expected in name for expected in expected_models) for name in names)
assert len(set(names)) == 1 and all(torch.cuda.get_device_capability(i) >= (8, 0)
                                   for i in range(expected_count))
assert version("torch").split("+")[0] == "2.7.1" and torch.version.cuda == "12.6"
for name, expected in {"transformers": "4.52.3", "trl": "0.15.2", "peft": "0.15.2", "accelerate": "1.7.0", "scipy": "1.15.3"}.items():
    assert version(name) == expected, (name, version(name))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print("GPU smoke passed:", expected_count, names[0])
