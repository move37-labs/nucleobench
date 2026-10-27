"""Compatibility boundary for differentiating local AlphaGenome predictions."""

import functools

import numpy as np
import torch


class JaxSelectedProfile(torch.autograd.Function):
    """First-order PyTorch VJP for a JAX function returning selected profiles."""

    @staticmethod
    def forward(ctx, x, function):
        import jax
        import jax.numpy as jnp

        array = jnp.asarray(x.detach().to(torch.float32).cpu().numpy())
        # Save only the input; recomputing the VJP avoids retaining trunk activations.
        ctx.function = function
        ctx.save_for_backward(x.detach())
        result = function(array)
        return torch.from_numpy(np.array(jax.device_get(result), copy=True)).to(
            x.device
        )

    @staticmethod
    def backward(ctx, grad_output):
        import jax
        import jax.numpy as jnp

        (x,) = ctx.saved_tensors
        array = jnp.asarray(x.to(torch.float32).cpu().numpy())
        _, vjp = jax.vjp(ctx.function, array)
        gradient = vjp(jnp.asarray(grad_output.detach().cpu().numpy()))[0]
        return torch.from_numpy(np.array(jax.device_get(gradient), copy=True)).to(
            device=x.device, dtype=x.dtype
        ), None


def selected_rna_profile(model, x, track_index):
    """Only entry point into AlphaGenome's private JAX prediction state."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import jmp
    from alphagenome.models.v1 import dna_model as dna_api
    from alphagenome.models.v1 import dna_output
    from alphagenome_research.model import dna_model as research_dna_model
    from alphagenome_research.model import model as research_model

    organism = dna_api.Organism.MUS_MUSCULUS
    metadata = model._metadata[organism]
    organism_index = jnp.full((x.shape[0],), 1, dtype=jnp.int32)

    if not hasattr(model, "_nucleobench_remat_predict"):
        policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

        @hk.transform_with_state
        def forward(sequence, index):
            with hk.mixed_precision.push_policy(research_model.AlphaGenome, policy):
                return research_model.AlphaGenome(
                    model._metadata,
                    remat=True,
                    freeze_trunk_embeddings=False,
                )(sequence, index)

        def apply(params, state, sequence, index):
            (predictions, _), _ = forward.apply(params, state, None, sequence, index)
            return predictions

        model._nucleobench_remat_predict = jax.jit(
            functools.partial(research_dna_model._predict, apply_fn=apply),
            static_argnames=["requested_outputs"],
        )

    def run(sequence):
        predictions = model._nucleobench_remat_predict(
            model._params,
            model._state,
            sequence,
            organism_index,
            requested_outputs=(dna_output.OutputType.RNA_SEQ,),
            negative_strand_mask=jnp.zeros((sequence.shape[0],), dtype=bool),
            strand_reindexing=metadata.strand_reindexing,
        )
        return jnp.asarray(
            predictions[dna_output.OutputType.RNA_SEQ][:, :, track_index],
            dtype=jnp.float32,
        )

    # Recompute the trunk in backward instead of retaining its activations.
    return jax.checkpoint(run)(jnp.transpose(x, (0, 2, 1)))
