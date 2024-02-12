import jax
import json
from jax.experimental import multihost_utils
from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform


def barrier(comm=None):
    if device == 'cpu':
        comm.Barrier()
    elif device == 'gpu':
        multihost_utils.sync_global_devices("barrier")
    else:
        raise ValueError("Unknown device type")

def start_step(step_name,comm=None):
    barrier(comm)
    curr_id = jax.process_index()
    if curr_id == 0:
        print("*"*77)  # total number of accelerator devices in the cluster
        print("*"*77)  # total number of accelerator devices in the cluster
        print(f"STEP {step_name} HAS BEGUN")  # total number of accelerator devices in the cluster
        print("*"*77)  # total number of accelerator devices in the cluster
        print("*"*77)  # total number of accelerator devices in the cluster
    barrier(comm)

def get_attribute_values(obj):
    attribute_values = {}
    attributes = dir(obj)
    for attr in attributes:
        try:
            attribute_values[attr] = str(getattr(obj, attr))
        except Exception as e:
            attribute_values[attr] = str(e)
    return attribute_values

def pretty_print(attributes_values):
    print(json.dumps(attributes_values, indent=4))

def inspect_attr(var, var_name, name):
    curr_id = jax.process_index()
    print("{0}{2} - {3}{1}".format("*"*77,"*"*77,name,var_name))
    print(f"Process {curr_id} : {name} => {var_name} : {var} \n")
    print(f"Process {curr_id} : {name} => {var_name} Type : {type(var)} \n")
    pretty_print(get_attribute_values(var))
    print("\n")

def print_array_info(a , name,comm=None):
    barrier(comm)
    curr_id = jax.process_index()
    if a.sharding.is_fully_addressable:
        inspect_attr(a, "data", name)
    inspect_attr(a.sharding, "sharding", name)
    inspect_attr(a.shape, "shape", name)
    inspect_attr(a.itemsize, "itemsize", name)
    inspect_attr(a.ndim, "ndim", name)
    inspect_attr(a.sharding.is_fully_addressable, "is_fully_addressable", name)
    inspect_attr(a.sharding.is_fully_replicated, "is_fully_replicated", name)
    inspect_attr(a.global_shards, "global_shards", name)
    inspect_attr(a.addressable_shards, "addressable_shards", name)
    inspect_attr(a.global_shards[0], "global_shards[0]", name)
    inspect_attr(a.addressable_shards[0], "addressable_shards[0]", name)
    inspect_attr(a.global_shards[0].data, "global_shards[0].data", name)

    inspect_attr(a.addressable_data, "addressable_data", name)
    inspect_attr(a.addressable_data(0), "addressable_data[0]", name)

    inspect_attr(a.addressable_shards[0].data, "addressable_shards[0].data", name)
    inspect_attr(a.global_shards[0]._sharding, "global_shards[0]._sharding", name)
    inspect_attr(a.addressable_shards[0]._sharding, "addressable_shards[0]._sharding", name)
    inspect_attr(a.addressable_shards[0].data.shape, "addressable_shards[0].data.shape", name)
    # This is none <<
    #print(f"Process {curr_id} : {name} global_shards data shape {a.global_shards[0].data.shape} \n\n")
    # This is none >>
    inspect_attr(a.global_shards[0]._global_shape, "global_shards[0]._global_shape", name)
    inspect_attr(a.addressable_shards[0]._global_shape, "addressable_shards[0]._global_shape", name)

    barrier(comm)