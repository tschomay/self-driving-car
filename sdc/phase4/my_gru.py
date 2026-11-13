"""The owner's GRU memory for the jumping car (LEARNER-OWNED, Phase 4 E3). See AGENTS.md.

Why memory: in the air the accelerometer reads about zero (free fall), so the car can't feel
which way is down. Its pitch has to be pieced together over time from the gyro (how fast it is
rotating) and what it knew at takeoff. A network that only sees the current sensor reading can't
do that sum; a recurrent one carries a memory vector h from one step to the next.

How a GRU (gated recurrent unit) works, one line per idea:
  - It keeps a memory vector h (say 64 numbers). Each step it reads the new input x and the old h.
  - A candidate memory  h~ = tanh(W x + U (r * h) + b)  is what the memory "would become" given x.
  - The update gate     z = sigmoid(Wz x + Uz h + bz)  (0..1 per number) decides how much of each
    memory number to replace with the candidate: h_new = (1 - z) * h + z * h~.
    z near 0 = keep the old memory (e.g. hold "pitch at takeoff"); near 1 = overwrite it.
  - The reset gate      r = sigmoid(Wr x + Ur h + br)  decides how much of the old memory the
    candidate may look at (r near 0 = compute the candidate from the input alone).
  - Because the new memory is a blend of the old one, information (and gradients) can survive
    many steps, which is what plain recurrent nets fail at.

OWNER TODOs below; the agent wrote the plumbing (rollout storage with memory, sequence
minibatches, evaluation). Each TODO currently returns None, so training refuses to start.
"""

from __future__ import annotations

import torch
from torch import nn


def gru_step(x, h, p):
    """OWNER TODO 1: one GRU step.

    x: (batch, n_in) input this step; h: (batch, n_hidden) memory from the last step.
    p: a dict of parameters: p["Wz"], p["Wr"], p["Wh"] (n_in -> n_hidden), p["Uz"], p["Ur"], p["Uh"]
       (n_hidden -> n_hidden), each an nn.Linear (call it like p["Wz"](x)); the W's carry the biases.
    Return the new memory, (batch, n_hidden).

    Owner's choice (2026-11-13): (b) blend with an update gate, plus a reset gate r. "B is the best
    blend where older memories are kept but down weighted. The gate allows a cutoff so the down
    weighted doesn't have to be too strong, but there can be a cut off so the rest of the lap
    doesn't impact this jump for example."
    """
    z = torch.sigmoid(p["Wz"](x) + p["Uz"](h))          # update gate: how much of each number to overwrite
    r = torch.sigmoid(p["Wr"](x) + p["Ur"](h))          # reset gate: how much old memory the candidate sees
    cand = torch.tanh(p["Wh"](x) + p["Uh"](r * h))      # candidate memory from this step's input
    return (1 - z) * h + z * cand                        # blend: keep (z~0) or overwrite (z~1)


def reset_memory(h, episode_started):
    """OWNER TODO 2: what happens to the memory when an episode ends and a fresh one starts.

    h: (batch, n_hidden); episode_started: (batch,) float, 1.0 where that car just began a new course.
    Return the memory to use for this step.
    """
    return None


CONFIG = dict(
    # OWNER TODO 3: shape of the memory network (None = not decided yet)
    hidden=None,            # memory size, e.g. 32, 64, 128
    where=None,             # "before": obs -> GRU -> actor/critic MLP heads; "beside": MLP heads see obs AND memory
    shared=None,            # True: one memory for actor and critic; False: each has its own
    # OWNER TODO 4: training on sequences
    seq_len=None,           # steps per training chunk (how far back gradients flow), e.g. 16, 32, 64
)


class GRUCell(nn.Module):
    """Holds the parameters for gru_step (plumbing)."""

    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.p = nn.ModuleDict(dict(Wz=nn.Linear(n_in, n_hidden), Wr=nn.Linear(n_in, n_hidden),
                                    Wh=nn.Linear(n_in, n_hidden),
                                    Uz=nn.Linear(n_hidden, n_hidden, bias=False),
                                    Ur=nn.Linear(n_hidden, n_hidden, bias=False),
                                    Uh=nn.Linear(n_hidden, n_hidden, bias=False)))
        self.n_hidden = n_hidden

    def forward(self, x, h):
        return gru_step(x, h, self.p)


def ready() -> bool:
    import torch as _t
    h = _t.zeros(2, 4)
    cell = GRUCell(3, 4)
    ok = gru_step(_t.zeros(2, 3), h, cell.p) is not None and reset_memory(h, _t.zeros(2)) is not None
    return ok and all(v is not None for v in CONFIG.values())
