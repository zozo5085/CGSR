"""Context-Guided Selective Rollback (CGSR): a training-free post-refinement verifier.

Code ownership:

* **CGSR-owned** (``cgsr/verifier.py``; depends only on PyTorch):
  ``SelectiveRollbackVerifier`` and ``local_class_support``. Importable on its
  own, without the base-predictor checkout or the candidate generator::

      from cgsr import SelectiveRollbackVerifier, local_class_support

* **Experimental integration** (``cgsr/model_cgsr.py``): ``CGSRPipeline``.
* **Third-party / upstream** (not redistributed; provenance in
  THIRD_PARTY_NOTICES.md): ``FrozenBasePredictor`` is the upstream ReCLIP++
  base predictor class ``RECLIPPP``, loaded from the user's checkout by
  ``cgsr.upstream.activate_base_code``. ``FixedCandidateGenerator`` is the class
  ``RECLIPPP`` of the separately distributed ``cgsr/candidate/model_sfp_dtlr.py``.

Pipeline::

    image -> FrozenBasePredictor      -> base_logits, projected_features, penultimate_attention
          -> FixedCandidateGenerator  -> candidate_logits
          -> SelectiveRollbackVerifier -> final_logits
          -> shared_postprocess       -> segmentation   (cgsr/evaluation.py)
"""

__all__ = ["SelectiveRollbackVerifier", "local_class_support"]


def __getattr__(name):
    # Lazy export (PEP 562): importing the package stays lightweight for the
    # command-line tools; the verifier needs only PyTorch.
    if name in __all__:
        from . import verifier

        return getattr(verifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
