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


def split_array2d(arr, row_splits, col_splits):
    """Split an array into a custom number of row and column splits."""
    row_height = arr.shape[0] // row_splits
    col_width = arr.shape[1] // col_splits
    result = []
    for r in range(row_splits):
        for c in range(col_splits):
            start_row = r * row_height
            end_row = (r + 1) * row_height
            start_col = c * col_width
            end_col = (c + 1) * col_width
            result.append(arr[start_row:end_row, start_col:end_col])
    return result

def split_array(arr, splits, axis=0):
    """Split an array into equal parts along a specified axis."""
    if len(splits) == 1:
        splits = splits[0]

    if isinstance(splits, int):
        return np.array_split(arr, splits, axis=axis)
    elif isinstance(splits,tuple):
        return split_array2d(arr, splits[0], splits[1])

def create_and_scatter_numpy_array(func:callable,split_shape, **kwargs):
    array_shape = None
    array_split = None
    if rank == 0:
        # Allocate the array on the root process
        array = func(**kwargs)
        array_shape = array.shape
        array_split = split_array(array,split_shape )

    # Split the array into 8 pieces on the root process
    array_piece = comm.scatter(array_split, root=0)
    # Copy array global shape to all processes
    array_shape = comm.bcast(array_shape, root=0)

    return array_piece


sharding_shape = (4,2)
# Create a generator object
rng = np.random.default_rng(12345)  # 12345 is the seed
# Generate random numbers
array = rng.normal(size=(128,128))
input_shape = array.shape

rng = np.random.default_rng(12345)  # 12345 is the seed
array_piece = create_and_scatter_numpy_array(rng.normal,split_shape=sharding_shape , size=(128, 128))

# Scatter the pieces of the array to all processes
print(f"Process {rank} has array piece {array_piece.shape}")
# Now, array_piece contains a piece of the array on each process

def cb(index):
    assert(not False in (array_piece == array[index]))
    return array_piece

jax.distributed.initialize()

partition_tuple = tuple(letter for letter in 'abcd'[:len(sharding_shape)])
devices = mesh_utils.create_device_mesh(sharding_shape)
global_mesh = Mesh(devices, axis_names=partition_tuple)
inp_sharding = jax.sharding.NamedSharding(global_mesh,P(*partition_tuple))

arr = jax.make_array_from_callback(input_shape, inp_sharding, cb)
if jax.process_index() == 1:
    print_array_info(arr, "arr")
    
barrier(device)
jax.distributed.shutdown()
