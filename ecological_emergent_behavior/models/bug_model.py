import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_data, static_functions
from mechagogue.nn.layer import standardize_layer, make_layer
from mechagogue.nn.linear import linear_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.utils import num_parameters
from mechagogue.ecology.policy import make_ecology_population
from mechagogue.tree import tree_getitem, tree_setitem
from dirt.gridworld2d.distributed import _make_perms
from mechagogue.breed.normal import normal_mutate

from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.envs.tera_arium import TeraAriumTraits
import dirt.bug as bug

from ecological_emergent_behavior.models.encoder import make_encoder
from ecological_emergent_behavior.models.newborn import do_nothing_if_newborn


@static_data
class BugModelParams:
    # printing/logging
    verbose : bool = True
    
    # max population
    max_players : int = 32
    
    # vision hyperparameters
    max_view_width : int = 15
    max_view_distance : int = 7
    max_view_back_distance : int = 7
    zero_vision : bool = False
    
    # nonvision hyperparameters
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    include_wind : bool = True
    include_temperature : bool = True
    include_audio : bool = True
    audio_channels : int = 8
    include_smell : bool = True
    smell_channels : int = 8
    include_compass : bool = True
    vision_includes_rgb : bool = True
    vision_includes_relative_altitude : bool = True
    
    # high level network settings
    include_encoder : bool = True
    include_backbone : bool = True
    include_nonvision_encoder : bool = True
    include_vision_encoder : bool = True
    vision_encoder_mode : str = 'flatten' # flatten | conv | attention
    backbone_mode : str = 'mlp' # linear | mlp
    learnable_temperature : bool = False
    
    # network hyperparameters
    hidden_channels : int = 256
    backbone_layers : int = 8
    use_bias : bool = True
    
    # mode specific settings
    # - conv/attention vision encoder
    conv_channels : int = 64
    # - attention vision encoder
    attention_mode : str = 'soft' # soft | hard
    
    # weight mutation rate and scaling
    base_mutation_likelihood : float = 1.
    base_mutation_rate : float = 3e-4
    base_mutation_density : float = None
    weight_mutation_likelihood : float = None
    weight_mutation_rate : float = None
    weight_mutation_density : float = None
    weight_decay_mode : str = 'constant'

    # policy migration settings
    # 0 disables chunking; otherwise chunk size must divide max_players
    policy_transfer_chunk : int = 0
    weight_decay : float = 0.
    mutate_traits : bool = True
    mutate_sensor_noise : bool = False
    mutate_color : bool = False
    mutation_auto_scale : bool = False
    sensors_start_noisy : bool = False
    
    # traits
    default_senescence_damage : float = None
    big_hit : bool = False
    
    def validate(params):
        # weights
        if params.weight_mutation_likelihood is None:
            params = params.replace(
                weight_mutation_likelihood =
                params.base_mutation_likelihood
            )
        if params.weight_mutation_rate is None:
            params = params.replace(
                weight_mutation_rate = params.base_mutation_rate)
        if params.weight_mutation_density is None:
            params = params.replace(
                weight_mutation_density = params.base_mutation_density)
        
        return params
    
    def vision_shape(params):
        return ( 
            params.max_view_distance + params.max_view_back_distance + 1, 
            params.max_view_width, 
        )


def make_bug_model(
    params,
    env,
    return_logging_info=False,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    params = params.validate()
    
    if params.verbose:
        print('model settings')
        print(f'  mutate traits: {params.mutate_traits}')
        print(f'  mutate sensors: {params.mutate_sensor_noise}')
        print(f'  include encoder: {params.include_encoder}')
    
    # determine how many output channels the encoder should have
    if not params.include_encoder:
        encoder_out_channels = 1
    elif params.include_backbone:
        encoder_out_channels = params.hidden_channels
    else:
        encoder_out_channels = env.num_actions
    
    # build the encoder
    if params.include_encoder:
        encoder = make_encoder(
            params,
            encoder_out_channels,
            return_activations=return_logging_info,
            dtype=dtype,
        )
    else:
        encoder = make_layer(
            forward=lambda : jnp.ones(encoder_out_channels, dtype=dtype))
        encoder.mutate = lambda key, state : state
    
    # build the backbone
    if params.verbose:
        print(f'  include backbone: {params.include_backbone}')
        print(f'    {env.num_actions} output channels')
    if params.include_backbone:
        if params.include_encoder:
            backbone_in_channels = params.hidden_channels
        else:
            backbone_in_channels = 1
        
        if params.backbone_mode == 'linear':
            backbone = linear_layer(
                backbone_in_channels,
                env.num_actions,
                use_bias=params.use_bias,
                dtype=dtype,
            )
            if return_logging_info:
                def forward(x, state):
                    x = backbone(x, state)
                    return x, {'linear':x}
                return make_layer(backbone.init, forward)
            mutate = normal_mutate(
                params.weight_mutation_rate,
                update_density=params.weight_mutation_density,
                auto_scale=params.mutation_auto_scale,
            )
            backbone.mutate = staticmethod(mutate)
        
        elif params.backbone_mode == 'mlp':
            backbone = mlp(
                params.backbone_layers,
                in_channels=backbone_in_channels,
                hidden_channels=params.hidden_channels,
                out_channels=env.num_actions,
                use_bias=params.use_bias,
                return_activations=return_logging_info,
                dtype=dtype,
            )
            mutate = normal_mutate(
                params.weight_mutation_rate,
                update_density=params.weight_mutation_density,
                auto_scale=params.mutation_auto_scale
            )
            backbone.mutate = staticmethod(mutate)
        
        else:
            raise ValueError(f'Unknown backbone_mode "{params.backbone_mode}"')
    
    else:
        def forward(x):
            if return_logging_info:
                return x, {}
            else:
                return x
        
        backbone = make_layer(forward=forward)
        backbone.mutate = staticmethod(lambda key, state : state)
        backbone.state_statistics = staticmethod(lambda name, state : {})
    
    sampler = categorical_sampler_layer(include_entropy=return_logging_info)
    
    if params.learnable_temperature:
        def make_temperature_layer():
            def init(key):
                return jnp.ones((1,), dtype=dtype)
            
            def forward(key, x, state):
                return x / state
            
            def mutate(key, state):
                state = jnp.exp(
                    jnp.log(state) +
                    jrng.normal(key, state.shape, dtype=state.dtype) * params.base_mutation_rate
                )
                return state
            
            temperature_layer = make_layer(init=init, forward=forward)
            temperature_layer.mutate = staticmethod(mutate)
            
            return temperature_layer
        
        temperature_layer = make_temperature_layer()
        layers = [encoder, backbone, temperature_layer, sampler]
    
    else:
        layers = [encoder, backbone, sampler]
    
    base_model = layer_sequence(layers)
    
    if return_logging_info:
        def forward(key, x, state):
            ek, bk, tk, sk = jrng.split(key, 4)
            if params.learnable_temperature:
                x, ea = encoder.forward(ek, x, state.layer_states[0])
                x, ba = backbone.forward(bk, x, state.layer_states[1])
                x = temperature_layer.forward(tk, x, state.layer_states[2])
                x, h = sampler.forward(sk, x, state.layer_states[3])
            else:
                x, ea = encoder.forward(ek, x, state.layer_states[0])
                x, ba = backbone.forward(bk, x, state.layer_states[1])
                x, h = sampler.forward(sk, x, state.layer_states[2])
            activations = {'encoder':ea, 'backbone':ba}
            return x, h, activations
            
        model = do_nothing_if_newborn(
            make_layer(base_model.init, forward)
        )
    else:
        model = do_nothing_if_newborn(base_model)
    
    def mutate(key, state):
        if params.learnable_temperature:
            encoder_key, backbone_key, temperature_key = jrng.split(key, 3)
            encoder_state, backbone_state, temperature_state, sampler_state = (
                state.layer_states)
            encoder_state = encoder.mutate(encoder_key, encoder_state)
            backbone_state = backbone.mutate(backbone_key, backbone_state)
            temperature_state = temperature_layer.mutate(
                temperature_key, temperature_state)
            return state.replace(
                layer_states=(
                    encoder_state,
                    backbone_state,
                    temperature_state,
                    sampler_state,
                )
            )
        else:
            encoder_key, backbone_key = jrng.split(key, 2)
            encoder_state, backbone_state, sampler_state = state.layer_states
            encoder_state = encoder.mutate(encoder_key, encoder_state)
            backbone_state = backbone.mutate(backbone_key, backbone_state)
            return state.replace(
                layer_states=(encoder_state, backbone_state, sampler_state))
    model.mutate = staticmethod(mutate)
    
    def state_statistics(name, state):
        datapoint = {}
        datapoint.update(encoder.state_statistics(
            f'{name}encoder', state.layer_states[0]))
        datapoint.update(backbone.state_statistics(
            f'{name}backbone', state.layer_states[1]))
        if params.learnable_temperature:
            datapoint[f'{name}temperature'] = state.layer_states[2][0]
        return datapoint
    model.state_statistics = staticmethod(state_statistics)
    
    if params.verbose:
        example_key = jrng.key(0)
        abstract_key = jax.ShapeDtypeStruct(
            example_key.shape, example_key.dtype)
        abstract_shape = jax.eval_shape(model.init, abstract_key)
        n = num_parameters(abstract_shape)
        print(f'  {n} total parameters')
    
    return model


def make_bug_policy(
    params,
    env,
    return_logging_info=False,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    model = make_bug_model(
        params, env, return_logging_info=return_logging_info, dtype=dtype)
    
    @static_functions
    class BugPolicy:
        def init(key):
            model_state = model.init(key)
            traits = TeraAriumTraits.default(())
            if params.big_hit:
                traits = traits.replace(
                    attack_primitives=jnp.array([[2,0,1,1,10]], dtype=jnp.float32))
            if params.default_senescence_damage is not None:
                traits = traits.replace(
                    senescence_damage=params.default_senescence_damage)
            if params.sensors_start_noisy:
                traits = traits.replace(
                    age_sensor_noise=jnp.ones_like(traits.age_sensor_noise),
                    visual_sensor_noise=jnp.ones_like(
                        traits.visual_sensor_noise),
                    audio_sensor_noise=jnp.ones_like(
                        traits.audio_sensor_noise),
                    smell_sensor_noise=jnp.ones_like(
                        traits.smell_sensor_noise),
                    external_resource_sensor_noise=jnp.ones_like(
                        traits.external_resource_sensor_noise),
                    wind_sensor_noise=jnp.ones_like(
                        traits.wind_sensor_noise),
                    temperature_sensor_noise=jnp.ones_like(
                        traits.temperature_sensor_noise),
                    compass_sensor_noise=jnp.ones_like(
                        traits.compass_sensor_noise),
                    health_sensor_noise=jnp.ones_like(
                        traits.health_sensor_noise),
                    internal_resource_sensor_noise=jnp.ones_like(
                        traits.internal_resource_sensor_noise),
                )
            return model_state, traits
        
        def act(key, obs, state):
            model_state, traits = state
            return model.forward(key, obs, model_state)
        
        def traits(state):
            return state[1]
        
        def mutate(key, state):
            return model.mutate(key, state)
        
        def state_statistics(name, state):
            return model.state_statistics(name, state)
    
    return BugPolicy


def make_bug_population(
    params,
    env,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    policy = make_bug_policy(params, env, dtype=dtype)
    log_policy = make_bug_policy(
        params.replace(verbose=False),
        env,
        return_logging_info=True,
        dtype=dtype,
    )
    def breed(key, parent_state):
        parent_model_state, parent_traits = parent_state
            
        # mutate model
        parent_model_state = tree_getitem(parent_model_state, 0)
        model_key, traits_key = jrng.split(key)
        child_model_state = policy.mutate(model_key, parent_model_state)
        
        # mutate traits
        parent_traits = tree_getitem(parent_traits, 0)
        if params.mutate_traits:
            child_traits = env.mutate_traits(traits_key, parent_traits)
        else:
            child_traits = parent_traits
            if params.mutate_sensor_noise:
                if params.include_vision_encoder:
                    traits_key, vision_key = jrng.split(traits_key)
                    child_traits = env.normal_mutate_trait(
                        vision_key, child_traits, 'visual_sensor_noise')
                if params.include_nonvision_encoder:
                    if params.include_compass:
                        traits_key, compass_key = jrng.split(traits_key)
                        child_traits = env.normal_mutate_trait(
                            compass_key, child_traits, 'compass_sensor_noise')
                    traits_key, internal_key, external_key = jrng.split(
                        traits_key, 3)
                    child_traits = env.normal_mutate_trait(
                        internal_key,
                        child_traits,
                        'internal_resource_sensor_noise',
                    )
                    child_traits = env.normal_mutate_trait(
                        external_key,
                        child_traits,
                        'external_resource_sensor_noise',
                    )
            if params.mutate_color:
                traits_key, color_key = jrng.split(traits_key)
                child_traits = env.normal_mutate_trait(
                    color_key, child_traits, 'color')
        
        return child_model_state, child_traits
    
    population = make_ecology_population(
        policy,
        params.max_players,
        breed,
    )

    def _set_members_masked(state, index, values, mask):
        def leaf_set(leaf, leaf_value):
            updated = leaf.at[index].set(leaf_value)
            mask_shape = (mask.shape[0],) + (1,) * (leaf.ndim - 1)
            return jnp.where(mask.reshape(mask_shape), updated, leaf)
        return jax.tree.map(leaf_set, state, values)

    def migrate(state, migrations):
        if migrations is None:
            return state
        src, dst = migrations
        if src is None or dst is None:
            return state
        if not getattr(env, "distributed", False):
            return state
        tile_dimensions = getattr(env, "tile_dimensions", (1, 1))
        if tile_dimensions == (1, 1):
            return state

        perms = _make_perms(*tile_dimensions)
        directions = [
            "up",
            "down",
            "left",
            "right",
            "up_left",
            "up_right",
            "down_left",
            "down_right",
        ]

        def ppermute_tree(tree, direction):
            return jax.tree.map(
                lambda leaf: jax.lax.ppermute(
                    leaf, axis_name="mesh", perm=perms[direction]),
                tree,
            )

        chunk = params.policy_transfer_chunk
        if (
            chunk is None
            or chunk <= 0
            or chunk >= params.max_players
            or (params.max_players % chunk) != 0
        ):
            for i, direction in enumerate(directions):
                src_idx = src[i]
                dst_idx = dst[i]
                valid = dst_idx >= 0
                if valid.ndim == 0:
                    valid = jnp.expand_dims(valid, 0)
                any_valid = jnp.any(valid)

                def apply_dir(s):
                    safe_src = jnp.where(valid, src_idx, 0)
                    safe_dst = jnp.where(valid, dst_idx, 0)
                    payload_state = ppermute_tree(s, direction)
                    moved = tree_getitem(payload_state, safe_src)
                    return _set_members_masked(s, safe_dst, moved, valid)

                state = jax.lax.cond(any_valid, apply_dir, lambda s: s, state)
            return state

        def slice_tree(tree, start, size):
            def leaf_slice(leaf):
                return jax.lax.dynamic_slice(
                    leaf,
                    (start,) + (0,) * (leaf.ndim - 1),
                    (size,) + leaf.shape[1:],
                )
            return jax.tree.map(leaf_slice, tree)

        num_chunks = params.max_players // chunk
        for i, direction in enumerate(directions):
            src_idx = src[i]
            dst_idx = dst[i]
            valid = dst_idx >= 0
            if valid.ndim == 0:
                valid = jnp.expand_dims(valid, 0)
            any_valid = jnp.any(valid)

            def apply_dir(s):
                for c in range(num_chunks):
                    start = c * chunk
                    end = start + chunk
                    chunk_mask = (
                        valid &
                        (src_idx >= start) &
                        (src_idx < end)
                    )
                    any_chunk = jnp.any(chunk_mask)

                    def apply_chunk(ss):
                        payload_chunk = ppermute_tree(
                            slice_tree(ss, start, chunk), direction)
                        safe_src = jnp.where(chunk_mask, src_idx - start, 0)
                        safe_dst = jnp.where(chunk_mask, dst_idx, 0)
                        moved = tree_getitem(payload_chunk, safe_src)
                        return _set_members_masked(
                            ss, safe_dst, moved, chunk_mask)

                    s = jax.lax.cond(any_chunk, apply_chunk, lambda ss: ss, s)
                return s

            state = jax.lax.cond(any_valid, apply_dir, lambda s: s, state)

        return state

    population.migrate = staticmethod(migrate)
    
    def log(key, state, obs, active):
        keys = jrng.split(key, params.max_players)
        actions, entropy, activations = jax.vmap(log_policy.act)(
            keys, obs, state)
        num_active = jnp.sum(active, dtype=jnp.int32)
        entropy_mean = jnp.sum(entropy*active, dtype=jnp.float32) / num_active
        datapoint = {'action/entropy':entropy_mean}
        executed_action_type = env.action_to_primitive[actions][...,0]
        for action_type in range(bug.NUM_ACTION_TYPES):
            if env.action_primitive_count[action_type] == 0:
                continue
            
            action_count = jnp.sum(
                (executed_action_type == action_type) * active,
                dtype=jnp.float32,
            )
            action_fraction = action_count / num_active
            action_type_name = bug.action_type_names[action_type].replace(
                ' ', '_')
            label = f'action/{action_type_name}'
            datapoint[label] = action_fraction
        
        path_activations, _ = jax.tree_util.tree_flatten_with_path(activations)
        for path, activation in path_activations:
            def get_path_name(item):
                if hasattr(item, 'key'):
                    return str(item.key)
                elif hasattr(item, 'idx'):
                    return str(item.idx)
                else:
                    return str(item)
            leaf_name = '_'.join(get_path_name(p) for p in path)
            label = f'activation/{leaf_name}.magnitude'
            magnitude = jnp.linalg.norm(activation, axis=-1)
            c = activation.shape[-1]
            magnitude /= c**0.5
            magnitude = magnitude.reshape(params.max_players, -1).mean(axis=-1)
            magnitude_sum = jnp.sum(magnitude * active, dtype=jnp.float32)
            magnitude_mean = magnitude_sum / num_active
            datapoint[label] = magnitude_mean
            
            mean_label = f'activation/{leaf_name}.mean'
            mean_activations = activation.reshape(
                params.max_players, -1).mean(axis=-1)
            datapoint[mean_label] = jnp.sum(
                mean_activations * active, dtype=jnp.float32) / num_active
            
            positive_label = f'activation/{leaf_name}.positive'
            positive = ((activation > 0.).sum(axis=-1) / c)
            positive = jnp.sum(positive * active, dtype=jnp.float32)/num_active
            datapoint[positive_label] = positive
        
        model_state, traits = state
        stats = jax.vmap(policy.state_statistics, in_axes=(None,0))(
            'model/', model_state)
        def compute_mean(leaf):
            return jnp.sum(leaf * active, dtype=jnp.float32) / num_active
        
        datapoint.update(jax.tree.map(compute_mean, stats))
        
        return datapoint
    
    population.log = staticmethod(log)
    
    return population


# bug model preset examples
BlindModelParams = BugModelParams(
    include_encoder = False,
    backbone_mode = 'linear',
)

LinearModelParams = BugModelParams(
    include_backbone = False,
    vision_encoder_mode = 'flatten',
)

FlattenMLPModelParams = BugModelParams(
    vision_encoder_mode = 'flatten',
    backbone_mode = 'mlp',
)

ConvMLPModelParams = BugModelParams(
    vision_encoder_mode = 'conv',
    backbone_mode = 'mlp',
)
