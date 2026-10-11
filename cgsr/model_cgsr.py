"""CGSR integration layer: frozen base predictor -> fixed candidate generator -> selective rollback verifier.

Role names used in this repository (provenance in THIRD_PARTY_NOTICES.md):

* ``FrozenBasePredictor``: the base predictor class of the upstream ReCLIP++
  code (Wang & Kang), imported unmodified from the user's checkout. It
  produces the base logits and the projected patch features.
* ``FixedCandidateGenerator``: the existing SFP + DTLR refinement stage
  (``cgsr/candidate/``, distributed separately). It subclasses the base
  predictor and produces the candidate logits.
* ``SelectiveRollbackVerifier`` / ``local_class_support``: the CGSR operator
  (this work), defined in ``cgsr/verifier.py`` and importable on its own.
* ``CGSRPipeline``: the experimental integration of base, candidate and
  verifier in one module (this file).

The pipeline deliberately performs the same two model passes as the frozen
research runtime (``run_p1.py``, an unreleased research script; the
"run_p1.py lines ..." comments below point into it and are kept for provenance
only):

1. ``return_feat=True`` for the base logits and the projected patch features.
2. Plain inference for the candidate logits.

CGSR has no learned parameters and does not consume ground truth or cached
attention. The cached penultimate attention remains an internal input to the
candidate generator only.
"""

from model.model import RECLIPPP as FrozenBasePredictor  # upstream base predictor class, unmodified
from cgsr.candidate import FixedCandidateGenerator
from .verifier import SelectiveRollbackVerifier, local_class_support  # noqa: F401  (CGSR-owned core)

assert issubclass(FixedCandidateGenerator, FrozenBasePredictor)


def _detached(tensor):
    """Return a detached snapshot without changing prediction tensors."""
    return None if tensor is None else tensor.detach().clone()


class CGSRPipeline(FixedCandidateGenerator):
    """Frozen base predictor + fixed candidate generator + selective rollback verifier.

    Normal test-time inference returns ``final_logits`` with shape
    ``(1,K,H,W)``, matching the shared evaluator's logits interface. When
    ``return_debug=True``, the return value is ``(final_logits, states)``,
    where ``states`` holds the base, candidate and verifier tensors. Training
    and explicit ``return_feat=True`` preserve the inherited return contracts.

    The frozen publication runtime uses batch size one. The inherited
    ``return_feat`` API itself removes the batch dimension, so the pipeline
    explicitly enforces that same batch-size-one inference contract.
    """

    verifier = SelectiveRollbackVerifier()

    def forward(
        self,
        image,
        gt_cls,
        zeroshot_weights,
        cls_name_token,
        training=False,
        img_metas=None,
        return_feat=False,
        return_debug=False,
    ):
        if training or return_feat:
            if return_debug:
                raise ValueError("return_debug is only supported for plain test-time inference")
            return super().forward(
                image,
                gt_cls,
                zeroshot_weights,
                cls_name_token,
                training=training,
                img_metas=img_metas,
                return_feat=return_feat,
            )

        if image.shape[0] != 1:
            raise ValueError(
                "CGSRPipeline reproduces the frozen batch-size-one publication runtime; "
                f"received batch size {image.shape[0]}"
            )

        # Frozen run_p1.py lines 179-185: independent base and candidate passes.
        base_logits, projected_features, _grid_shape = super().forward(
            image,
            gt_cls,
            zeroshot_weights,
            cls_name_token,
            training=False,
            img_metas=img_metas,
            return_feat=True,
        )
        base_logits = base_logits.float()
        projected_features = projected_features.float()

        candidate = super().forward(
            image,
            gt_cls,
            zeroshot_weights,
            cls_name_token,
            training=False,
            img_metas=img_metas,
            return_feat=False,
        )
        candidate_logits = candidate[0].float()
        penultimate_attention = self._sfp_attn_cache

        final_logits, states = self.verifier(base_logits, candidate_logits, projected_features)
        output = final_logits.unsqueeze(0)

        if not return_debug:
            return output

        debug = {
            "base_logits": _detached(base_logits),
            "projected_features": _detached(projected_features),
            "penultimate_attention": _detached(penultimate_attention),
            "candidate_logits": _detached(candidate_logits),
            "final_logits": _detached(final_logits),
            "final_class": _detached(final_logits.argmax(0)),
        }
        debug.update({key: _detached(value) for key, value in states.items()})
        return output, debug
