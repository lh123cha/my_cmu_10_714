import sys
sys.path.append("../python")
import numpy as np
import needle as ndl
import needle.nn as nn

sys.path.append("../apps")
from mlp_resnet import *

import mugrade

t1 = ndl.Tensor([0.0])