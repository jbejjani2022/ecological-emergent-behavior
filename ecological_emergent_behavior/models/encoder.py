from typing import Any

import numpy as np

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static import static_data, static_functions
from mechagogue.nn.layer import make_layer, standardize_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.linear import linear_layer, conv_layer
from mechagogue.nn.attention import make_attention_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.debug import (
    print_shape_layer, print_activations_layer, breakpoint_layer)
from mechagogue.breed.normal import normal_mutate

from dirt.envs.tera_arium import TeraAriumTraits
from dirt.constants import DEFAULT_FLOAT_DTYPE

from ecological_emergent_behavior.models.mutable_linear import make_mutable_linear


def make_reshape_layer(output_shape):
    return make_layer(forward=lambda x : x.reshape(output_shape))


def make_nonvision_encoder(
    params,
    out_channels,
    return_activations=False,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    def make_encoder_part(in_channels, initial_value=0.):
        linear1 = make_mutable_linear(
            in_channels,
            out_channels,
            use_bias=params.use_bias,
            init_weights=zero, 
            mutation_likelihood=params.weight_mutation_likelihood,
            mutation_rate=params.weight_mutation_rate,
            update_density=params.weight_mutation_density,
            dtype=dtype,
        )
        
        def init(key):
            linear_state = linear1.init(key)
            return linear_state
        
        def forward(x, state):
            linear_state = state
            x = linear1.forward(x, linear_state)
            return x
        
        def mutate(key, state):
            linear_state = state
            key, linear_key = jrng.split(key)
            linear_state = linear1.mutate(linear_key, linear_state)
            
            return linear_state
        
        def state_statistics(name, state):
            stats = {}
            linear_state = state
            stats.update(linear1.state_statistics(
                f'{name}_linear', linear_state))
            return stats
        
        model = make_layer(init=init, forward=forward)
        model.mutate = staticmethod(mutate)
        model.state_statistics = staticmethod(state_statistics)
        
        return model
    
    age_encoder_part = make_encoder_part(1, initial_value=1.)
    health_encoder_part = make_encoder_part(1)
    
    resource_channels = 0
    if params.include_water:
        resource_channels += 2
    if params.include_energy:
        resource_channels += 2
    if params.include_biomass:
        resource_channels += 2
    if resource_channels:
        has_resource_encoder = True
        resource_encoder_part = make_encoder_part(resource_channels)
    else:
        has_resource_encoder = False
    
    weather_channels = 0
    if params.include_wind:
        weather_channels += 2
    if params.include_temperature:
        weather_channels += 1
    if weather_channels:
        has_weather_encoder = True
        weather_encoder_part = make_encoder_part(weather_channels)
    else:
        has_weather_encoder = False
    
    if params.include_audio:
        audio_encoder_part = make_encoder_part(params.audio_channels)
    
    if params.include_smell:
        smell_encoder_part = make_encoder_part(params.smell_channels)
    
    if params.include_compass:
        compass_encoder_part = make_encoder_part(4)
    
    @static_data
    class NonvisualEncoderState:
        age_state : Any = None
        health_state : Any = None
        resource_state : Any = None
        weather_state : Any = None
        audio_state : Any = None
        smell_state : Any = None
        compass_state : Any = None
    
    @static_functions
    class NonvisualEncoder:
        def init(key):
            state = NonvisualEncoderState()
            
            key, age_key = jrng.split(key)
            state = state.replace(age_state=age_encoder_part.init(age_key))
            
            key, health_key = jrng.split(key)
            state = state.replace(
                health_state=health_encoder_part.init(health_key))
            
            if has_resource_encoder:
                key, resource_key = jrng.split(key)
                state = state.replace(
                    resource_state=resource_encoder_part.init(resource_key))
            
            if has_weather_encoder:
                key, weather_key = jrng.split(key)
                state = state.replace(
                    weather_state=weather_encoder_part.init(weather_key))
            
            if params.include_audio:
                key, audio_key = jrng.split(key)
                state = state.replace(
                    audio_state=audio_encoder_part.init(audio_key))
            
            if params.include_smell:
                key, smell_key = jrng.split(key)
                state = state.replace(
                    smell_state=smell_encoder_part.init(smell_key))
            
            if params.include_compass:
                key, compass_key = jrng.split(key)
                state = state.replace(
                    compass_state=compass_encoder_part.init(compass_key))
            
            return state
        
        def forward(x, state):
            features = []
            activations = {}
            
            features.append(age_encoder_part.forward(
                x.age[...,None].astype(dtype), state.age_state))
            features.append(health_encoder_part.forward(
                x.health[...,None].astype(dtype), state.health_state))
            
            if has_resource_encoder:
                resources = []
                if params.include_water:
                    resources.append(x.external_water)
                    resources.append(x.internal_water)
                if params.include_energy:
                    resources.append(x.external_energy)
                    resources.append(x.internal_energy)
                if params.include_biomass:
                    resources.append(x.external_biomass)
                    resources.append(x.internal_biomass)
                resource_x = jnp.stack(resources, axis=-1)
                features.append(resource_encoder_part.forward(
                    resource_x.astype(dtype), state.resource_state))
                
                if return_activations:
                    activations['resource_input'] = resource_x
                    activations['resource_output'] = features[-1]
            
            if has_weather_encoder:
                weather_features = []
                if params.include_wind:
                    weather_features.append(x.wind)
                if params.include_temperature:
                    weather_features.append(x.temperature[...,None])
                weather_x = jnp.concatenate(weather_features, axis=-1)
                features.append(weather_encoder_part.forward(
                    weather_x.astype(dtype), state.weather_state))
                
                if return_activations:
                    activations['weather_input'] = weather_x
                    activations['weather_output'] = features[-1]
            
            if params.include_audio:
                audio_x = x.audio.astype(dtype)
                features.append(audio_encoder_part.forward(
                    audio_x, state.audio_state))
                
                if return_activations:
                    activations['audio_input'] = audio_x
                    activations['audio_output'] = features[-1]
            
            if params.include_smell:
                smell_x = x.smell.astype(dtype)
                features.append(smell_encoder_part.forward(
                    smell_x, state.smell_state))
                
                if return_activations:
                    activations['smell_input'] = smell_x
                    activations['smell_output'] = features[-1]
            
            if params.include_compass:
                compass_x = x.compass.astype(dtype)
                features.append(compass_encoder_part.forward(
                    compass_x, state.compass_state))
                
                if return_activations:
                    activations['compass_input'] = compass_x
                    activations['compass_output'] = features[-1]
            
            x = sum(features)
            
            if return_activations:
                return x, activations
            else:
                return x
        
        def mutate(key, state):
            key, age_key = jrng.split(key)
            state = state.replace(age_state=age_encoder_part.mutate(
                age_key, state.age_state))
            
            key, health_key = jrng.split(key)
            state = state.replace(health_state=health_encoder_part.mutate(
                health_key, state.health_state))
            
            if has_resource_encoder:
                key, resource_key = jrng.split(key)
                state = state.replace(
                    resource_state=resource_encoder_part.mutate(
                        resource_key, state.resource_state))
            
            if has_weather_encoder:
                key, weather_key = jrng.split(key)
                state = state.replace(weather_state=weather_encoder_part.mutate(
                    weather_key, state.weather_state))
            
            if params.include_audio:
                key, audio_key = jrng.split(key)
                state = state.replace(audio_state=audio_encoder_part.mutate(
                    audio_key, state.audio_state))
            
            if params.include_smell:
                key, smell_key = jrng.split(key)
                state = state.replace(smell_state=smell_encoder_part.mutate(
                    smell_key, state.smell_state))
            
            if params.include_compass:
                key, compass_key = jrng.split(key)
                state = state.replace(compass_state=compass_encoder_part.mutate(
                    compass_key, state.compass_state))
            
            return state
        
        def state_statistics(name, state):
            stats = {}
            stats.update(age_encoder_part.state_statistics(
                f'{name}_age', state.age_state))
            
            stats.update(health_encoder_part.state_statistics(
                f'{name}_health', state.health_state))
            
            if has_resource_encoder:
                stats.update(resource_encoder_part.state_statistics(
                    f'{name}_resource', state.resource_state))
            
            if has_weather_encoder:
                stats.update(weather_encoder_part.state_statistics(
                    f'{name}_weather', state.weather_state))
            
            if params.include_audio:
                stats.update(audio_encoder_part.state_statistics(
                    f'{name}_audio', state.audio_state))
            
            if params.include_smell:
                stats.update(smell_encoder_part.state_statistics(
                    f'{name}_smell', state.smell_state))
            
            if params.include_compass:
                stats.update(compass_encoder_part.state_statistics(
                    f'{name}_compass', state.compass_state))
            
            return stats
            
    
    return NonvisualEncoder


def make_vision_extractor(
    params,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    if params.verbose:
        vision_shape = params.vision_shape()
        vision_channels = num_vision_channels(params)
        vision_total_channels = np.prod(vision_shape) * vision_channels
        print(
            f'      {vision_shape} vision shape with {vision_channels} '
            f'channels ({vision_total_channels} total input vision channels)'
        )
    def forward(x):
        vision_components = []
        if params.vision_includes_rgb:
            vision_components.append(x.rgb.astype(dtype))
        if params.vision_includes_relative_altitude:
            vision_components.append(
                x.relative_altitude[...,None].astype(dtype))
        vision_data = jnp.concatenate(vision_components, axis=-1)
        
        if params.zero_vision:
            vision_data = jnp.zeros_like(vision_data)
        
        return vision_data
    
    return make_layer(forward=forward)


def num_vision_channels(params):
    return (
        3 * params.vision_includes_rgb +
        1 * params.vision_includes_relative_altitude
    )


def make_flatten_vision_encoder(
    params,
    out_channels,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    vision_shape = params.vision_shape()
    in_channels = num_vision_channels(params)
    
    vision_extractor = make_vision_extractor(params, dtype=dtype)
    flattened_channels = vision_shape[0] * vision_shape[1] * in_channels
    flatten_layer = make_reshape_layer(flattened_channels)
    
    linear1 = make_mutable_linear(
        flattened_channels,
        out_channels,
        use_bias=params.use_bias,
        mutation_likelihood=params.weight_mutation_likelihood,
        mutation_rate=params.weight_mutation_rate,
        update_density=params.weight_mutation_density,
        dtype=dtype,
    )
    
    layers = [vision_extractor, flatten_layer, linear1]
    encoder = layer_sequence(layers)
    
    def mutate(key, state):
        key, linear_key = jrng.split(key)
        linear_state = linear1.mutate(linear_key, state.layer_states[2])
        return state.replace(layer_states=(None, None, linear_state))
    
    def state_statistics(name, state):
        stats = {}
        linear_state = state.layer_states[2]
        stats.update(linear1.state_statistics(f'{name}_linear', linear_state))
        
        return stats
    
    encoder.mutate = staticmethod(mutate)
    encoder.state_statistics = staticmethod(state_statistics)
    
    return encoder


def make_conv_flatten_vision_encoder(
    params,
    out_channels,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    
    vision_shape = params.vision_shape()
    assert vision_shape[0] % 3 == 0
    assert vision_shape[1] % 3 == 0
    in_channels = num_vision_channels(params)
    
    vision_extractor = make_vision_extractor(params, dtype=dtype)
    
    conv1 = conv_layer(
        in_channels,
        params.conv_channels,
        kernel_size=(3,3),
        stride=(3,3),
        padding='VALID',
        use_bias=params.use_bias,
        dtype=dtype,
    )
    
    post_conv_shape = (vision_shape[0] // 3, vision_shape[1] // 3)
    flattened_channels = (
        post_conv_shape[0] * post_conv_shape[1] * params.conv_channels)
    flatten1 = make_reshape_layer((flattened_channels,))
    
    relu1 = relu_layer()
    
    linear1 = linear_layer(
        flattened_channels,
        out_channels,
        use_bias=params.use_bias,
        dtype=dtype,
    )
    
    encoder = layer_sequence(
        (vision_extractor, conv1, flatten1, relu1, linear1))
    
    mutate = normal_mutate(
        params.weight_mutation_rate,
        update_density=params.weight_mutation_density,
        auto_scale=params.mutation_auto_scale,
    )
    encoder.mutate = staticmethod(mutate)
    
    def state_statistics(name, state):
        datapoint = {}
        datapoint.update(conv1.state_statistics(
            f'{name}_conv', state.layer_states[1]))
        datapoint.update(linear1.state_statistics(
            f'{name}_linear', state.layer_states[4]))
        return datapoint
    
    return encoder


def make_conv_attention_vision_encoder(
    params,
    out_channels,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    vision_shape = params.vision_shape()
    assert vision_shape[0] % 3 == 0
    assert vision_shape[1] % 3 == 0
    in_channels = num_vision_channels(params)
    
    vision_extractor = make_vision_extractor(params, dtype=dtype)
    
    tokens_h = vision_shape[0] // 3
    tokens_w = vision_shape[1] // 3
    total_tokens = tokens_h * tokens_w
    
    conv1 = conv_layer(
        in_channels,
        params.conv_channels,
        kernel_size=(3,3),
        stride=(3,3),
        padding='VALID',
        use_bias=params.use_bias,
        dtype=dtype,
    )
    
    kv_linear = linear_layer(
        params.conv_channels,
        params.conv_channels*2,
        use_bias=params.use_bias,
        dtype=dtype
    )
    
    attention1 = make_attention_layer(1., params.attention_mode)
    
    relu1 = relu_layer()
    
    linear1 = linear_layer(
        params.conv_channels,
        out_channels,
        use_bias=params.use_bias,
        dtype=dtype,
    )
    
    @static_data
    class ConvAttentionEncoderState:
        conv1 : Any
        kv_linear : Any
        position_embedding : Any
        q : Any
        linear1 : Any
    
    @static_functions
    class ConvAttentionEncoder:
        def init(key):
            conv1_key, kv_linear_key, pe_key, q_key, linear1_key = jrng.split(
                key, 5)
            conv1_state = conv1.init(conv1_key)
            kv_linear_state = kv_linear.init(kv_linear_key)
            position_embedding_state = jnp.zeros(
                (total_tokens, params.conv_channels))
            q_state = jrng.normal(
                q_key,
                (1, params.conv_channels)
            ) * (1./params.conv_channels)**0.5
            linear1_state = linear1.init(linear1_key)
            return ConvAttentionEncoderState(
                conv1=conv1_state,
                kv_linear=kv_linear_state,
                position_embedding=position_embedding_state,
                q=q_state,
                linear1=linear1_state,
            )
        
        def forward(key, x, state):
            x = vision_extractor.forward(x)
            x = conv1.forward(x, state.conv1)
            *b,h,w,c = x.shape
            x = x.reshape(*b,h*w,c)
            x = x + state.position_embedding
            kv = kv_linear.forward(x, state.kv_linear)
            k = kv[...,:params.conv_channels].reshape(-1, conv_channels)
            v = kv[...,params.conv_channels:].reshape(-1, conv_channels)
            
            x = attention1.forward(key, (state.q, k, v))[0]
            x = relu1.forward(x)
            x = linear1.forward(x, state.linear1)
            return x
    
        mutate = normal_mutate(
            params.weight_mutation_rate,
            update_density=params.weight_mutation_density,
            auto_scale=params.mutation_auto_scale,
        )
    
    return ConvAttentionEncoder


def make_encoder(
    params,
    out_channels,
    return_activations=False,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    if params.verbose:
        print(f'    include nonvision encoder {params.include_nonvision_encoder}')
        print(f'    include vision encoder {params.include_vision_encoder}')
    
    if params.include_nonvision_encoder:
        nonvision_encoder = make_nonvision_encoder(
            params,
            out_channels,
            return_activations=return_activations,
            dtype=dtype,
        )
        nonvision_encoder = standardize_layer(nonvision_encoder)
    
    if params.include_vision_encoder:
        if params.verbose:
            print(f'      mode "{params.vision_encoder_mode}"')
        if params.vision_encoder_mode == 'flatten':
            vision_encoder = make_flatten_vision_encoder(
                params, out_channels, dtype=dtype)
        elif params.vision_encoder_mode == 'conv':
            vision_encoder = make_conv_flatten_vision_encoder(
                params, out_channels, dtype=dtype)
        elif params.vision_encoder_mode == 'attention':
            vision_encoder = make_conv_attention_vision_encoder(
                params, out_channels, dtype=dtype)
        vision_encoder = standardize_layer(vision_encoder)
    
    @static_functions
    class Encoder:
        
        @static_data
        class EncoderState:
            if params.include_nonvision_encoder:
                nonvision_state : Any = None
            if params.include_vision_encoder:
                vision_state : Any = None
        
        def init(key):
            state = Encoder.EncoderState()
            if params.include_nonvision_encoder:
                key, nonvision_key = jrng.split(key)
                nonvision_state = nonvision_encoder.init(nonvision_key)
                state = state.replace(nonvision_state=nonvision_state)
            if params.include_vision_encoder:
                key, vision_key = jrng.split(key)
                vision_state = vision_encoder.init(vision_key)
                state = state.replace(
                    vision_state=vision_state,
                )
            return state
        
        def forward(key, x, state):
            x1 = jnp.zeros(out_channels)
            if return_activations:
                activations = {}
            if params.include_nonvision_encoder:
                key, nonvision_key = jrng.split(key)
                nonvision_x = nonvision_encoder.forward(
                    nonvision_key, x, state.nonvision_state)
                if return_activations:
                    nonvision_x, nonvision_activations = nonvision_x
                    activations['nonvision'] = nonvision_x
                    activations.update(nonvision_activations)
                x1 += nonvision_x
            if params.include_vision_encoder:
                key, vision_key = jrng.split(key)
                vision_x = vision_encoder.forward(
                    vision_key, x, state.vision_state)
                if return_activations:
                    activations['vision'] = vision_x
                x1 += vision_x
            
            if return_activations:
                activations['combined'] = x1
                return x1, activations
            else:
                return x1
        
        def mutate(key, state):
            next_state = Encoder.EncoderState()
            if params.include_nonvision_encoder:
                key, nonvision_key = jrng.split(key)
                nonvision_state = nonvision_encoder.mutate(
                    nonvision_key, state.nonvision_state)
                next_state = next_state.replace(nonvision_state=nonvision_state)
            if params.include_vision_encoder:
                key, vision_key = jrng.split(key)
                vision_state = vision_encoder.mutate(
                    vision_key, state.vision_state)
                next_state = next_state.replace(vision_state=vision_state)
            return next_state
        
        def state_statistics(name, state):
            datapoint = {}
            if params.include_nonvision_encoder:
                datapoint.update(nonvision_encoder.state_statistics(
                    f'{name}_nonvision', state.nonvision_state))
            if params.include_vision_encoder:
                datapoint.update(vision_encoder.state_statistics(
                    f'{name}_vision', state.vision_state))
            
            return datapoint
    
    return Encoder
