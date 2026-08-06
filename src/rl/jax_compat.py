"""jax 兼容层：补回 brax 需要但 jax 0.10 已移除的 API。

brax 0.14.2（最新）仍调用 jax.device_put_replicated，而 jax 0.10.2 把它移除了
（deprecated -> removed）。mujoco_playground 依赖 brax>=0.14.2，所以走它也是同一个坑。

好在 brax 的 PPO 训练里只有**一处**调用：
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:n])
语义是把 pytree 复制到 n 个设备上，得到前面多一维 (n, ...) 的数组。
jax.pmap 仍然可用，所以补上这一个函数就够，不必降级重装 2GB 的 CUDA 轮子。

import 本模块即完成打补丁，必须在 import brax 之前。
"""
import jax
import jax.numpy as jnp


def _device_put_replicated(x, devices):
    """把 pytree x 复制到每个 device，输出前置一维 len(devices)。

    等价于旧的 jax.device_put_replicated。单设备下就是加一维再放上去；
    多设备下用 pmap 广播，保证分片正确。
    """
    n = len(devices)
    if n == 1:
        return jax.tree.map(
            lambda y: jax.device_put(jnp.expand_dims(jnp.asarray(y), 0), devices[0]), x)
    # 多设备：用 pmap 的输出分片语义，天然是 (n, ...) 且已分片
    return jax.pmap(lambda _, y: y, axis_name="i",
                    in_axes=(0, None), devices=devices)(jnp.arange(n), x)


def _device_put_sharded(shards, devices):
    """把一列分片放到对应设备上，输出前置一维 len(shards)。"""
    return jax.tree.map(
        lambda *ys: jax.device_put(jnp.stack(ys), devices[0])
        if len(devices) == 1 else jnp.stack(ys), *shards)


def patch():
    applied = []
    if not hasattr(jax, "device_put_replicated"):
        jax.device_put_replicated = _device_put_replicated
        applied.append("device_put_replicated")
    if not hasattr(jax, "device_put_sharded"):
        jax.device_put_sharded = _device_put_sharded
        applied.append("device_put_sharded")
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree.map
        applied.append("tree_map")
    return applied


_APPLIED = patch()
