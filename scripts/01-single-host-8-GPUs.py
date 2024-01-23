import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Create a Sharding object to distribute a value across devices:
sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))

# Create an array of random values:
x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
jax.debug.visualize_array_sharding(x)
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, sharding.reshape(2, 2))
jax.debug.visualize_array_sharding(y)

x = jax.device_put(x, sharding.reshape(1,4))
jax.debug.visualize_array_sharding(x)


x = jax.device_put(x, sharding.reshape(4, 1))
jax.debug.visualize_array_sharding(x)