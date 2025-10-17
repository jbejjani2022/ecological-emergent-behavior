import jax.numpy as jnp

from mechagogue.nn.layer import make_layer

def do_nothing_if_newborn(model):
    def forward(key, x, state):
        output = model.forward(key, x, state)
        try:
            model_action, *other = output
        except:
            model_action = output
            other = None
        action = jnp.where(x.newborn, 0, model_action)
        if other is not None:
            return action, *other
        else:
            return action
    
    return make_layer(model.init, forward)
