
WIP
# introduction

In this tutorial, we delve into the world of distributed computing with JAX on the JeanZay supercalculator. We focus on the multi-controller model, a crucial concept for leveraging JAX's full potential on high-performance computing systems. This guide is tailored for users aiming to optimize their JAX applications for environments like JeanZay, where efficient distribution across multiple controllers is essential.

In JAX, each process runs independently in what's known as a Single Program, Multiple Data (SPMD) model. This differs from traditional distributed systems where a single node controls multiple workers. In JAX, each process runs a slightly varied version of the same Python program. For instance, different processes may handle distinct data segments. However, JAX requires manual execution on each host; it does not support automatic multi-process initiation from a single command.

When executing JAX code on a single machine, particularly with multiple GPUs, certain considerations and configurations are necessary. This section will guide you through setting up a single host with multiple GPUs.

---
<span style="font-size: 26px;">Table of content</span>
---


- [introduction](#introduction)
  - [Table of content](#table-of-content)
  - [Understanding Sharding in JAX](#understanding-sharding-in-jax)
    - [Sharding Properties](#sharding-properties)
    - [Special Kinds of Shards](#special-kinds-of-shards)
- [Single host multi GPU](#single-host-multi-gpu)
  - [Slurm](#slurm)
  - [Example](#example)
- [Multiple controller (one GPU per controller)](#multiple-controller-one-gpu-per-controller)
  - [Slurm](#slurm-1)
  - [Example](#example-1)
  - [Classic PositionalSharding in Multi-controller setup](#classic-positionalsharding-in-multi-controller-setup)
  - [Reshaping a GlobalDeviceShard](#reshaping-a-globaldeviceshard)
- [Bonus : Splitting, Slicing and Gathering data in a multi controller setup](#bonus--splitting-slicing-and-gathering-data-in-a-multi-controller-setup)



## Understanding Sharding in JAX

Sharding in JAX refers to how data is distributed across different devices in a multi-controller environment. It's essential to comprehend the properties and types of sharding available in JAX to utilize the hardware efficiently.

### Sharding Properties

1. [device_set](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Sharding.device_set): Represents the group of devices spanned by a specific sharding. In a multi-controller environment, this set includes devices that may not be directly addressable by the current process.
2. [is_fully_addressable](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Sharding.is_fully_addressable): Indicates whether all devices named in the Sharding are accessible by the current process. This property is analogous to is_local in a multi-process JAX setup.
3. [is_fully_replicated](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.SingleDeviceSharding.is_fully_replicated): Determines if each device holds a complete copy of the entire data. A sharding is fully replicated when this condition is met.
4. [shard_shape](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.XLACompatibleSharding.shard_shape)(global_shape): Get the shape of data on each device based on the global shape and the sharding's characteristics.

### Special Kinds of Shards

- SingleDeviceSharding: Data is placed on a single device (the default case where we place an entire copy of the data on GPU0).
- PositionalSharding: Uses device positioning for data distribution.

```python
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

You can find a detailed [example here](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#intro-and-a-quick-example) 


- NamedSharding: Employs named axes to express sharding.

```python
mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
spec = P('x', 'y')
named_sharding = jax.sharding.NamedSharding(mesh, spec)
```
You can find a detailed [example here ](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)

- PmapSharding: Matches the default placement used by jax.pmap(), it is the output of a pmapped function basically.
- GSPMDSharding: Considers a global set of devices, including those not directly addressable in a multi-controller setup we will see an example of this later on.

# Single host multi GPU

Refer to [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#) for detailed guidance.

JAX is always multicontroller from what I understood, but for simplicity I will call it single controller when there is one process launched by the end user (that's you) even if JAX launches multiple processes under the hood.

## Slurm

First we need to set up our `.slurm` script, you can find it [here](slurms/02-multihost-2nodes.slurm)


The most important arguments are
```
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
```

`nodes`, `ntasks` and `ntasks-per-node` are redundant together only one of `nodes`, `ntasks` is required.\
But notice here that I used only one task for 4 GPUs

## Example

This is the example from https://jax.readthedocs.io

```python
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Create a Sharding object to distribute a value across devices:
sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))

# Create an array of random values:
x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, sharding.reshape(4, 1))
jax.debug.visualize_array_sharding(y)

x = jax.device_put(x, sharding.reshape(2, 2))
jax.debug.visualize_array_sharding(x)

x = jax.device_put(x, sharding.reshape(1, 4))
jax.debug.visualize_array_sharding(x)
```

We can see how trivial it is to change the sharding shape to fit different distribution needs.\
Let's visualize the sharding of `x` we call `jax.debug.visualize_array_sharding` and get.


```
┌───────────────────────┐
│                       │
│                       │
│                       │
│                       │
│         GPU 0         │
│                       │
│                       │
│                       │
│                       │
└───────────────────────┘
```

Which makes sense, in a single controller multi device, when we allocate anything it is sent to the main GPU (the first one).\
when we reshape the sharding like this :

```python
y = jax.device_put(x, sharding.reshape(2, 2))
jax.debug.visualize_array_sharding(y)
```

We get the reshaped sharding accordingly 

```
┌──────────┬──────────┐
│          │          │
│  GPU 0   │  GPU 1   │
│          │          │
│          │          │
├──────────┼──────────┤
│          │          │
│  GPU 2   │  GPU 3   │
│          │          │
│          │          │
└──────────┴──────────┘
```

Then subsequently these two reshapes

```python
x = jax.device_put(x, sharding.reshape(1,4))
jax.debug.visualize_array_sharding(x)


x = jax.device_put(x, sharding.reshape(4, 1))
jax.debug.visualize_array_sharding(x)
```
produces these shardings

```
┌───────┬───────┬───────┬───────┐
│       │       │       │       │
│       │       │       │       │
│       │       │       │       │
│       │       │       │       │
│ GPU 0 │ GPU 1 │ GPU 2 │ GPU 3 │
│       │       │       │       │
│       │       │       │       │
│       │       │       │       │
│       │       │       │       │
└───────┴───────┴───────┴───────┘
┌───────────────────────┐
│         GPU 0         │
├───────────────────────┤
│         GPU 1         │
├───────────────────────┤
│         GPU 2         │
├───────────────────────┤
│         GPU 3         │
└───────────────────────┘
```

All of this is very trivial but there is a caveat, single controller (as in single process launched by you) is not scallable because it assumes that you are running on a single node. For JeanZay for example a V100 has quadri-core gpus and the A100 partition has octo-core, which means if we want to do anything on more that 8 GPUs this [Single host multi GPU](#single-host-multi-gpu) won't work.

# Multiple controller (one GPU per controller)

Single controller means single node, which means it is not scallable, for JeanZay for example a V100 has quadricore gpus and the A100 partition has octo-core, which means if we want to do anything on more that 8 GPUs this [Single host multi GPU](#single-host-multi-gpu) won't work.

## Slurm 

Just like [Single host multi GPU](#single-host-multi-gpu) we need to set up our `.slurm` script, you can find it [here](slurms/02-multihost-2nodes.slurm)

The most important arguments are
```
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
```
Notice that we still allocate 4 GPUs but there are 4 GPUs per node. In total we have 8.

## Example

To begin we need to call this before doing any computation with jax 

```python
jax.distributed.initialize()
```

for more info read [multi_process](https://jax.readthedocs.io/en/latest/multi_process.html).

__Note:__ for JeanZay, we do not need to specify any arguments for  `jax.distributed.initialize()` because everything is set up by `srun`


Next we can print `jax.device_count()` and `jax.local_device_count()` and we can see that they do not have the same count, as opposed to the single controller case.\
A very important change is the fact that every single line of code is being ran 8 times (in our case) in the previous case we had one controller, so every line ran once except for the `pmaped` function of course because `pmap` is an implicit parallel call

We can print a debug message only once like this :


```python
curr_id = jax.process_index()
if curr_id == 0:
    print(f"Total number of devices {jax.device_count()}")
```

Now if we create any jax array types just like before 


```python
# The psum is performed over all mapped devices across the pod slice
xs = jax.numpy.ones(jax.local_device_count())
jax.debug.visualize_array_sharding(xs)
```

When we visualize the shardings we see the sharding layout 8 times for each gpus like so :


```
┌───────────────┐
│     GPU 1     │
└───────────────┘
┌───────────────┐
│     GPU 0     │
└───────────────┘
┌───────────────┐
│     GPU 3     │
└───────────────┘
┌───────────────┐
│     GPU 2     │
└───────────────┘
┌───────────────┐
│     GPU 7     │
└───────────────┘
┌───────────────┐
│     GPU 6     │
└───────────────┘
┌───────────────┐
│     GPU 4     │
└───────────────┘
┌───────────────┐
│     GPU 5     │
└───────────────┘
```

__Note:__ order is never guarenteed in multi controller calls, the only way to ensure that two steps do not interfer with each other is be using [sync_global_devices](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.multihost_utils.sync_global_devices.html#jax.experimental.multihost_utils.sync_global_devices) which is similar to `MPI_Barrier` for those who are familiar with MPI


If we check if the sharding is adressable we get `True`, our 8 part arrays are also `fully_replicated` which we mentionend in [Sharding Properties](#sharding-properties) which means that each device has a complete copy of the entire data which is not our case, so what is really happening.

In case of multi controller each code is called n times as we mentionned before, so this line 

```
xs = jax.numpy.ones(jax.local_device_count())
```

Is called n times and each of the devices is allocating a slice of the data that is fully addressable and fully replicable for each controller, but they are not globally.\
In other words each process can access it's data so it is adressable for this given process and it has a full copy of its data so it's replicable for this given process.

What does this mean for sharding reshape like we did earler?

## Classic PositionalSharding in Multi-controller setup

```python
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
```

Does not work and it produces this error 


```
PositionalSharding error: device_put's second argument must be a Device or a Sharding which represents addressable devices, but got PositionalSharding([[{GPU 0} {GPU 1}]
                    [{GPU 2} {GPU 3}]
                    [{GPU 4} {GPU 5}]
                    [{GPU 6} {GPU 7}]]). You are probably trying to use device_put in multi-controller JAX which is not supported. Please use jax.make_array_from_single_device_arrays API or pass device or Sharding which represents addressable devices.
```

This means that each device can do reshaping, but for it's addressable shards only, in our case we have 8 GPUs 8 process which means we have only SingleDeviceShard, we can't reshape a scaler.

the error from the jax team suggested to do [make_array_from_single_device_arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.make_array_from_single_device_arrays.html)

Following their example : 


```python
import math
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

mesh_rows = 2
mesh_cols =  jax.device_count() // 2

global_shape = (8, 8)
mesh = Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
inp_data = np.arange(math.prod(global_shape)).reshape(global_shape)

arrays = [
   jax.device_put(inp_data[index], d)
       for d, index in sharding.addressable_devices_indices_map(global_shape).items()]

arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
assert arr.shape == (8,8) # arr.shape is (8,8) regardless of jax.device_count()
```

We see that we can load the entire data to the CPU before dispatching it according to the suggested sharding to the gpus.\
This is a bottleneck for scallability as it suggests that our data can fit on CPU memory or RAM.

A different approach would be to try to recombine the SingleDeviceShards to get the full picture, this is done using this code


```python
global_mesh = jax.sharding.Mesh(jax.devices(), 'x')
pspecs = jax.sharding.PartitionSpec('x')
host_id = jax.process_index()
arr = multihost_utils.host_local_array_to_global_array(xs, global_mesh, pspecs)
```

If we visualize the array sharding, we see this 8 times : 


```
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ GPU 0 │ GPU 1 │ GPU 2 │ GPU 3 │ GPU 4 │ GPU 5 │ GPU 6 │ GPU 7 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

Now this makes sense, this array is also not fully_addressable nor fully_replicable, which means that in term of memory layout we did nothing, we only mapped the position of other shards for all controllers.\
But that does not mean that we can access them, the only way to do stuff to non adressable shards is via [lax collectives](https://jax.readthedocs.io/en/latest/jax.lax.html#parallel-operators) 

If we try to print the array we get this error :

```
Print global array error: Fetching value for `jax.Array` that spans non-addressable devices is not possible. You can use `jax.experimental.multihost_utils.process_allgather` for this use case.
```

Makes sense, since the code is called locally 8 times, and none of the 8 processes can see the other shards unless a communication is established via a collective call.

We can follow the advice of the JAX team and use `process_allgather`

```python
arr_gathered = multihost_utils.process_allgather(arr)
print(f"Process {curr_id} : Gathered Global array value is {arr_gathered}")
```

This works, but if we check the type of the `arr_gathered` it is a numpy (thus cpu) array.\
In general printing the `GlobalDeviceShard` does not make sense anyway

## Reshaping a GlobalDeviceShard

If we want to do a special kind of layout for pencil distribution for example like so 

```python
devices = mesh_utils.create_device_mesh((4,2))
mesh = Mesh(devices, axis_names=('a', 'b'))
pspecs = PartitionSpec('a', 'b')
arr4_2 = multihost_utils.host_local_array_to_global_array(xs, mesh, pspecs)
```

Just make sure that your array is divisible by b for example :

```python
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
```


# Bonus : Splitting, Slicing and Gathering data in a multi controller setup

