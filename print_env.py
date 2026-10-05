"""Print library versions, to record the tested environment."""
import platform

import numpy
import pandas
import sklearn

print("python", platform.python_version())
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scikit-learn", sklearn.__version__)
try:
    import torch
    import torchvision
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    print("torchvision", torchvision.__version__)
except ImportError:
    print("torch / torchvision not installed")
