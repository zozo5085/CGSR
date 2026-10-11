"""CGSR core: 5x5 local class support and the selective rollback verifier.

This module is the CGSR-owned contribution. It depends only on PyTorch: it does
not import the base predictor (upstream ReCLIP++), the candidate generator,
configs, checkpoints or the evaluation code, so it can be imported and used on
its own::

    from cgsr import SelectiveRollbackVerifier, local_class_support

``SelectiveRollbackVerifier()(base_logits, candidate_logits, projected_features)``
takes token-grid tensors ``(K,Hp,Wp)``, ``(K,Hp,Wp)`` and ``(C,Hp,Wp)`` and
returns ``(final_logits, states)``. The experimental integration with the frozen
base predictor and the fixed candidate generator is ``cgsr/model_cgsr.py``.

The "run_p1.py lines ..." comment points into the unreleased research script
that held the frozen runtime; it is kept for provenance only.
"""

import torch
import torch.nn.functional as F

__all__ = ["local_class_support", "SelectiveRollbackVerifier"]


def local_class_support(P, feat):
    """Frozen CGSR context builder: 5x5 local class support.

    ``P`` is ``(K,H,W)`` (base class probabilities) and ``feat`` is ``(C,H,W)``
    (unit-norm projected features). The 5x5 neighbourhood includes the centre
    and uses ``-inf`` affinity masking at truncated borders.
    """
    K, H, W = P.shape
    feat_pad = F.pad(feat, (2, 2, 2, 2), mode="constant", value=0.0)
    P_pad = F.pad(P, (2, 2, 2, 2), mode="constant", value=0.0)
    valid_pad = F.pad(torch.ones(1, H, W, device=P.device), (2, 2, 2, 2), value=0.0)
    affs, Ps = [], []
    for di in range(-2, 3):
        for dj in range(-2, 3):
            sf = feat_pad[:, 2 + di:2 + di + H, 2 + dj:2 + dj + W]
            sv = valid_pad[:, 2 + di:2 + di + H, 2 + dj:2 + dj + W]
            sp = P_pad[:, 2 + di:2 + di + H, 2 + dj:2 + dj + W]
            a = (feat * sf).sum(0)
            a = a.masked_fill(sv[0] == 0, float("-inf"))
            affs.append(a)
            Ps.append(sp)
    aff = torch.stack(affs, 0)
    w = torch.softmax(aff, dim=0)
    Pstack = torch.stack(Ps, 0)
    return (w.unsqueeze(1) * Pstack).sum(0)


class SelectiveRollbackVerifier:
    """CGSR operator: keep the candidate where its class has more local support.

    Frozen settings: 5x5 window, temperature 1, strict ``g > 0`` gate,
    whole-logit selection at the native token resolution. No parameters.
    """

    def __call__(self, base_logits, candidate_logits, projected_features):
        """``(K,Hp,Wp)``, ``(K,Hp,Wp)``, ``(C,Hp,Wp)`` -> ``final_logits (K,Hp,Wp)``, states."""
        # Frozen run_p1.py lines 188-198: shared context, strict gate, rollback.
        base_probs = torch.softmax(base_logits, dim=0)
        local_support = local_class_support(base_probs, projected_features)
        base_class = base_probs.argmax(0)
        candidate_class = torch.softmax(candidate_logits, dim=0).argmax(0)
        support_base = local_support.gather(0, base_class.unsqueeze(0)).squeeze(0)
        support_candidate = local_support.gather(0, candidate_class.unsqueeze(0)).squeeze(0)
        support_gain = support_candidate - support_base
        accept = support_candidate > support_base
        C = base_logits.shape[0]
        final_logits = torch.where(accept.unsqueeze(0).expand(C, -1, -1), candidate_logits, base_logits)
        states = {
            "base_probs": base_probs,
            "local_support": local_support,
            "base_class": base_class,
            "candidate_class": candidate_class,
            "support_base": support_base,
            "support_candidate": support_candidate,
            "support_gain": support_gain,
            "accept": accept,
        }
        return final_logits, states
