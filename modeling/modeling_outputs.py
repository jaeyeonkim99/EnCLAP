from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class EnClapBartOutput(Seq2SeqLMOutput):
    mcm_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
