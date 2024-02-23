import contextlib

import torch
import torch.nn as nn

class BioMedGPTBase(nn.Module):
    def __init__(self):
        super(BioMedGPTBase, self).__init__()

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_pretrained(model_name_or_path):
        raise NotImplementedError