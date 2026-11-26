"""Evaluation against real editing pipelines and held-out models.

Held-out models used here must never be added to soteria.surrogates.REGISTRY:
that separation is what makes the transfer results honest.
"""
