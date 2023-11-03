import jax
import flax
import optax

from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
from flax import linen as nn
from flax import serialization

# Dense layer instance (taking 'features' parameter as input)
model = nn.Dense(features=5)

key1, key2 = random.split(random.key(0))
x = random.normal(key1, (10,)) # Dummy input data
params = model.init(key2, x) # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params) # Checking output shapes

model.apply(params, x)

# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.key(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# Store the parameters in a FrozenDict pytree.
true_params = flax.core.freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)

# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
  # Define the squared loss for a single pair (x,y)
  def squared_error(x, y):
    pred = model.apply(params, x)
    return jnp.inner(y-pred, y-pred) / 2.0
  # Vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)

learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', mse(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)

@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params

for i in range(101):
  # Perform one gradient update.
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  params = update_params(params, learning_rate, grads)
  if i % 10 == 0:
    print(f'Loss step {i}: ', loss_val)

tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

for i in range(101):
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)

bytes_output = serialization.to_bytes(params)
dict_output = serialization.to_state_dict(params)
print('Dict output')
print(dict_output)
print('Bytes output')
print(bytes_output)

serialization.from_bytes(params, bytes_output)


class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print('output:\n', y)

try:
    y = model(x) # Returns an error
except AttributeError as e:
    print(e)


class SimpleMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      # providing a name is optional though!
      # the default autonames would be "Dense_0", "Dense_1", ...
    return x

key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print('output:\n', y)


class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.features))  # shape info.
    y = jnp.dot(inputs, kernel)
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameters:\n', params)
print('output:\n', y)


class BiasAdderWithRunningMean(nn.Module):
  decay: float = 0.99

  @nn.compact
  def __call__(self, x):
    # easy pattern to detect if we're initializing via empty variable tree
    is_initialized = self.has_variable('batch_stats', 'mean')
    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s),
                            x.shape[1:])
    bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
    if is_initialized:
      ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)

    return x - ra_mean.value + bias


key1, key2 = random.split(random.key(0), 2)
x = jnp.ones((10,5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print('initialized variables:\n', variables)
y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
print('updated state:\n', updated_state)

for val in [1.0, 2.0, 3.0]:
  x = val * jnp.ones((10,5))
  y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
  old_state, params = flax.core.pop(variables, 'params')
  variables = flax.core.freeze({'params': params, **updated_state})
  print('updated state:\n', updated_state) # Shows only the mutable part


  from functools import partial

@partial(jax.jit, static_argnums=(0, 1))
def update_step(tx, apply_fn, x, opt_state, params, state):

  def loss(params):
    y, updated_state = apply_fn({'params': params, **state},
                                x, mutable=list(state.keys()))
    l = ((x - y) ** 2).sum()
    return l, updated_state

  (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return opt_state, params, state

x = jnp.ones((10,5))
variables = model.init(random.key(0), x)
state, params = flax.core.pop(variables, 'params')
del variables
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for _ in range(3):
  opt_state, params, state = update_step(tx, model.apply, x, opt_state, params, state)
  print('Updated state: ', state)
