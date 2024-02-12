import math
import numpy as np
import jax
from jax import config, numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.lib import xla_bridge
from jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding
from jax.sharding import PartitionSpec as P
from mpi4py import MPI
from jax.lib import xla_bridge
import json
import numpy as np
from mpi4py import MPI
from ..utils import barrier, start_step, inspect_attr, print_array_info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# make random numpy array with shape input_shape
array = np.random.normal(size=(128, 128))
input_shape = array.shape

def cb(index):
    return array[index]

jax.distributed.initialize()

devices = mesh_utils.create_device_mesh((4,2))
global_mesh = Mesh(devices, axis_names=('a', 'b'))
inp_sharding = jax.sharding.NamedSharding(global_mesh,P('a', 'b'))

start_step("Create array")

arr = jax.make_array_from_callback(input_shape, inp_sharding, cb)
if jax.process_index() == 1:
    print_array_info(arr, "arr")

start_step("Transpose array")

transposed_sharding = jax.sharding.NamedSharding(global_mesh,P('b', 'a'))
transposed_arr = jax.make_array_from_callback(input_shape, transposed_sharding, cb)
if jax.process_index() == 1:
    print_array_info(transposed_arr, "transposed_arr")

    
barrier(device)
jax.distributed.shutdown()