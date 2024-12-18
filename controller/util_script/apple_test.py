import torch
print("torch.version",torch.version.__version__)

print("torch.cuda.is_available()",torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.current_device()",torch.cuda.current_device())

import torch_geometric
print("torch_geometric.__version__",torch_geometric.__version__)

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")