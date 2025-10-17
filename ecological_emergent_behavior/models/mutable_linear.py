import jax.numpy as jnp
import jax.random as jrng

from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.linear import linear_layer
from dirt.constants import DEFAULT_FLOAT_DTYPE


def make_mutable_linear(
    in_channels,
    out_channels,
    use_weight=True,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    mutation_likelihood=1.,
    mutation_rate=3e-2,
    update_density=None,
    weight_decay_mode='constant',
    weight_decay=0.,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    linear1 = linear_layer(
        in_channels,
        out_channels,
        use_weight=use_weight,
        use_bias=use_bias,
        init_weights=init_weights,
        init_bias=init_bias,
        dtype=dtype)
    
    def mutate(key, state):
        key, mutate_key = jrng.split(key)
        do_mutate = jrng.bernoulli(mutate_key, mutation_likelihood)
        if use_weight:
            key, weight_key, mask_key = jrng.split(key, 3)
            weight_delta = jrng.normal(
                weight_key,
                shape=state.weight.shape,
                dtype=state.weight.dtype,
            ) * mutation_rate
            if update_density is not None:
                delta_mask = jrng.bernoulli(
                    mask_key, update_density, weight_delta.shape)
                weight_delta =  weight_delta * delta_mask
            if weight_decay_mode == 'auto':
                raise NotImplementedError
            weight_delta += -state.weight * weight_decay
            weight = jnp.where(
                do_mutate, state.weight + weight_delta, state.weight)

            
            state = state.replace(weight=weight)
        
        if use_bias:
            key, bias_key = jrng.split(key)
            bias_delta = jrng.normal(
                bias_key,
                shape=state.bias.shape,
                dtype=state.bias.dtype,
            ) * mutation_rate
            bias = jnp.where(do_mutate, state.bias + bias_delta, state.bias)
            
            state = state.replace(bias=bias)
        
        return state
    
    linear1.mutate = staticmethod(mutate)
    
    return linear1
