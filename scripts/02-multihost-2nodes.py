# The following is run in parallel on each host on a GPU cluster or TPU pod slice.
import jax
from jax.experimental import multihost_utils
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


def print_section(name):
    multihost_utils.sync_global_devices("sync")

    curr_id = jax.process_index()
    if curr_id == 0:
        print("*"*77)
        print("*"*77)
        print(name)  
        print("*"*77) 
        print("*"*77)

    multihost_utils.sync_global_devices("sync")



jax.distributed.initialize()  # On GPU, see above for the necessary arguments.

curr_id = jax.process_index()
if curr_id == 0:
    print(f"Total number of devices {jax.device_count()}")  # total number of accelerator devices in the cluster

multihost_utils.sync_global_devices("sync")
print(f"Process {curr_id} : Devices in current process {jax.local_device_count()}")
multihost_utils.sync_global_devices("sync")

# The psum is performed over all mapped devices across the pod slice
xs = jax.numpy.ones(jax.local_device_count())
jax.debug.visualize_array_sharding(xs)

multihost_utils.sync_global_devices("arrsharding")
print(f"Process {curr_id} : xs array is_fully_addressable {xs.sharding.is_fully_addressable}")
print(f"Process {curr_id} : xs array is_fully_replicable {xs.sharding.is_fully_replicable}")
multihost_utils.sync_global_devices("arrsharding")

print(f"Process {curr_id} : Sharding of XS is  {xs.sharding}")
multihost_utils.sync_global_devices("sync")

y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print(f"Process {curr_id} : Value of of y is  {y}")
multihost_utils.sync_global_devices("sync")

print(f"Process {curr_id} : Sharding of y is  {y.sharding}")
multihost_utils.sync_global_devices("sync")

print_section("PositionalSharding in multi controller setup")

try:
    sharding = PositionalSharding(mesh_utils.create_device_mesh((jax.device_count(),)))
    # Create an array of random values:
    x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
    jax.debug.visualize_array_sharding(x)
    multihost_utils.sync_global_devices("sync")
    # and use jax.device_put to distribute it across devices:
    y = jax.device_put(x, sharding.reshape(4, 2))
    #jax.debug.visualize_array_sharding(y)
except Exception as e:
    print(f"PositionalSharding error: {e}")

multihost_utils.sync_global_devices("sync")

print_section("GlobalDeviceArray from SingleDeviceSharding")

global_mesh = jax.sharding.Mesh(jax.devices(), 'x')
pspecs = jax.sharding.PartitionSpec('x')
host_id = jax.process_index()
arr = multihost_utils.host_local_array_to_global_array(xs, global_mesh, pspecs)  #

print(f"Process {curr_id} : Sharding of array {arr.sharding}")
multihost_utils.sync_global_devices("arrsharding")
jax.debug.visualize_array_sharding(arr)
multihost_utils.sync_global_devices("arrsharding")

try:
    print(f"Process {curr_id} : Global array value is {arr}")
except Exception as e:
    print(f" Global array print error: {e}")
multihost_utils.sync_global_devices("arrsharding")

arr_gathered = multihost_utils.process_allgather(arr)

print(f"Process {curr_id} : Gathered Global array value is {arr_gathered}")
multihost_utils.sync_global_devices("arrsharding")
#jax.debug.visualize_array_sharding(arr_gathered)
#multihost_utils.sync_global_devices("arrsharding")

try:
    print(f"Process {curr_id} : Global array type {type(arr)}")
    print(f"Process {curr_id} : Global array is_fully_addressable {arr.sharding.is_fully_addressable}")
    multihost_utils.sync_global_devices("arrsharding")
except Exception as e:
    print(f" Global array print error: {e}")

try:
    print(f"Process {curr_id} : Gathered Global array type {type(arr_gathered)}")
    print(f"Process {curr_id} : Gathered Global array is_fully_addressable {arr_gathered.sharding.is_fully_addressable}")
    multihost_utils.sync_global_devices("arrsharding")
except Exception as e:
    print(f" Global array print error: {e}")

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.experimental import mesh_utils


print_section("Special Sharding")

devices = mesh_utils.create_device_mesh((4,2))
mesh = Mesh(devices, axis_names=('a', 'b'))
pspecs = PartitionSpec('a', 'b')
arr4_2 = multihost_utils.host_local_array_to_global_array(xs, mesh, pspecs)  #


multihost_utils.sync_global_devices("sync")
if curr_id == 0:
    print(devices)
multihost_utils.sync_global_devices("sync")

try:
    jax.debug.visualize_array_sharding(arr)
except Exception as e:
    print(f" visualize_array_sharding arr4_2 error: {e}")

multihost_utils.sync_global_devices("arrsharding")


# pspecs2_4 = PartitionSpec('b', 'a')
# arr2_4= multihost_utils.host_local_array_to_global_array(xs, mesh, pspecs)  #

# try:
#     jax.debug.visualize_array_sharding(arr)
# except Exception as e:
#     print(f" visualize_array_sharding arr2_4 error: {e}")


