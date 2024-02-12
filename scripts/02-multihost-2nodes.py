from jax import config
from jax.experimental import mesh_utils, multihost_utils
from jax.lib import xla_bridge
from jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import json
from utils import barrier, start_step, inspect_attr, print_array_info

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#jax.config.update('jax_platform_name', 'cpu')
jax.distributed.initialize()  # On GPU, see above for the necessary arguments.

device = xla_bridge.get_backend().platform

curr_id = jax.process_index()
if curr_id == 1:
    print(f"Total number of devices {jax.device_count()}")  # total number of accelerator devices in the cluster
    print(f"devices {jax.devices()}")  # total number of accelerator devices in the cluster

barrier(comm)
print(f"Process {curr_id} : Devices in current process {jax.local_device_count()}")
barrier(comm)

# The psum is performed over all mapped devices across the pod slice
#xs = jax.numpy.ones(jax.local_device_count())
key = jax.random.PRNGKey(0)
subkeys = jax.random.split(key, jax.device_count())
xs = jax.random.normal(subkeys[curr_id],(128,128))
jax.debug.visualize_array_sharding(xs,use_color=False)


start_step("Check XS")
if curr_id == 1:
    print_array_info(xs , "xs")
    
#xs_8 = xs.reshape(jax.device_count(), xs.shape[0] // jax.device_count(), *xs.shape[1:])
#y = jax.pmap(lambda x: print(x))(jnp.expand_dims(xs, 0))
#print(f"Process {curr_id} : Value of of y is  {y}")
#barrier(comm)

#print(f"Process {curr_id} : Sharding of y is  {y.sharding}")
barrier(comm)
try_classic_sharding = False
if try_classic_sharding == True:

    start_step("Classic sharding does not work")
    try:
        sharding = PositionalSharding(mesh_utils.create_device_mesh((jax.device_count(),)))
        # Create an array of random values:
        x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
        jax.debug.visualize_array_sharding(x,use_color=False)
        barrier(comm)
        # and use jax.device_put to distribute it across devices:
        y = jax.device_put(x, sharding.reshape(4, 2))
        #jax.debug.visualize_array_sharding(y,use_color=False)
    except Exception as e:
        print(f"PositionalSharding error: {e}")


    start_step("Try 8 1 host_local_array_to_global_array")




xs = jax.random.normal(subkeys[curr_id],(128,128))
global_mesh = jax.sharding.Mesh(jax.devices(), 'x')
pspecs = jax.sharding.PartitionSpec('x')
host_id = jax.process_index()
arr = multihost_utils.host_local_array_to_global_array(xs, global_mesh, pspecs)  #

jax.jit(jaxdecomp.fft)(arr)

1 - finir mon custom

if curr_id == 1:
    print_array_info(arr ,"arr")

def fun_sum(x):
    return jnp.sum(x,axis=0)

summed_arr = fun_sum(arr)

if curr_id == 1:
    print_array_info(summed_arr ,"summed_arr")


barrier(comm)
jax.debug.visualize_array_sharding(arr,use_color=False)
barrier(comm)

start_step("Cannot print global array")


if try_classic_sharding == True:
    try:
        print(f"Process {curr_id} : Global array value is {arr}")
    except Exception as e:
        print(f" Global array print error: {e}")
    barrier(comm)

    start_step("Can print Gathered CPU Array")

    print(f"Process {curr_id} : Gathered Global array value is {arr_gathered}")
    barrier(comm)
#jax.debug.visualize_array_sharding(arr_gathered)
##multihost_utils.sync_global_devices("arrsharding")

    try:
        print(f"Process {curr_id} : Global array type {type(arr)}")
        print(f"Process {curr_id} : Global array is_fully_addressable {arr.sharding.is_fully_addressable}")
        #multihost_utils.sync_global_devices("next")
    except Exception as e:
        print(f" Global array print error: {e}")

    try:
        print(f"Process {curr_id} : Gathered Global array type {type(arr_gathered)}")
        print(f"Process {curr_id} : Gathered Global array is_fully_addressable {arr_gathered.sharding.is_fully_addressable}")
        barrier(comm)
    except Exception as e:
        print(f" Global array print error: {e}")

start_step(" Testing Pencil setups 4 2")

devices = mesh_utils.create_device_mesh((4,2))

barrier(comm)
if curr_id == 1:
    print(devices)
barrier(comm)

mesh = Mesh(devices, axis_names=('a', 'b'))
pspecs = PartitionSpec('b', 'a')
arr4_2 = multihost_utils.host_local_array_to_global_array(xs, mesh, pspecs) 
sharding = NamedSharding(mesh,pspecs)

if rank == 0:
    print(inspect_attr(mesh, "mesh", "Standalone"))
    print(inspect_attr(pspecs, "pspecs", "Standalone"))
    print(inspect_attr(sharding, "sharding", "Standalone"))

    # verify if mesh is same as sharding.mesh
    print(f"Standalone : pspecs is same as sharding.pspecs : {pspecs == sharding.spec} \n")
    # verifu if pspecs is same as sharding.pspecs 
    print(f"Standalone : mesh is same as sharding.mesh : {mesh == sharding.mesh} \n")

    print(f"Standalone : Checking size of pspecs : {len(pspecs)} \n")
    print(f"Standalone : First element should be b : {pspecs[0]} \n")
    print(f"Standalone : Second element should be a : {pspecs[1]} \n")

    print(f"Standlone : first axis size : {mesh.shape[pspecs[0]]} \n")
    print(f"Standlone : second axis size : {mesh.shape[pspecs[1]]} \n")

    # make tuple shape out of pspecs and mesh
    shape = tuple(mesh.shape[axis] for axis in pspecs)

    print(f"Standalone : shape : {shape} \n")

barrier(comm)
barrier(comm)


if rank == 0:
    print_array_info(arr4_2 ,"arr4_2")

try:
    jax.debug.visualize_array_sharding(arr4_2,use_color=False)
except Exception as e:
    print(f" visualize_array_sharding arr4_2 error: {e}")

barrier(comm)
start_step(" Testing Pencil setups 2 4")

#new_arr = jax.pmap(lambda x: jax.lax.psum(x, 'a'),axis_name='a')(xs)
#if curr_id == 1:
   # print_array_info(new_arr ,"new_arr")


pspecs2_4 = PartitionSpec('b', 'a')
arr2_4= multihost_utils.host_local_array_to_global_array(xs, mesh, pspecs2_4)  #
if rank == 0:
    print_array_info(arr2_4 ,"arr2_4")

try:
     jax.debug.visualize_array_sharding(arr2_4,use_color=False)
except Exception as e:
     print(f" visualize_array_sharding arr2_4 error: {e}")

pspecs2_4 = PartitionSpec('b', 'a')
arr2_4= multihost_utils.host_local_array_to_global_array(xs, mesh, pspecs2_4)  #
if rank == 0:
    print_array_info(arr2_4 ,"arr2_4")

barrier(comm)
#jax.distributed.shutdown()
