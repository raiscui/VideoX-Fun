import importlib.util
import os

import torch
import torch.distributed as dist

try:
    # The pai_fuser is an internally developed acceleration package, which can be used on PAI.
    if importlib.util.find_spec("paifuser") is not None:
        import paifuser
        from paifuser.xfuser.core.distributed import (
            get_sequence_parallel_rank, get_sequence_parallel_world_size,
            get_sp_group, get_world_group, init_distributed_environment,
            initialize_model_parallel, model_parallel_is_initialized)
        from paifuser.xfuser.core.long_ctx_attention import \
            xFuserLongContextAttention
        print("Import PAI DiT Turbo")
    else:
        import xfuser
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group, get_world_group,
                                             init_distributed_environment,
                                             initialize_model_parallel,
                                             model_parallel_is_initialized)
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        print("Xfuser import sucessful")
except Exception as ex:
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    xFuserLongContextAttention = None
    get_sp_group = None
    get_world_group = None
    init_distributed_environment = None
    initialize_model_parallel = None


def _get_local_rank() -> int:
    """读取 torchrun 注入的 LOCAL_RANK."""

    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_local_world_size(default_world_size: int) -> int:
    """读取当前节点的本地 worker 数.

    优先使用 torchrun 注入的 `LOCAL_WORLD_SIZE`.
    如果环境变量缺失, 再退回调用方期望的 worker 数.
    """

    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is None:
        return default_world_size
    return int(local_world_size)


def _validate_local_cuda_topology(expected_local_workers: int) -> int:
    """在真正初始化 FSDP / NCCL 前, 先确认本地 rank 可映射到真实 GPU.

    这样能把 `invalid device ordinal` 这种深层 CUDA 异常,
    提前收敛成更容易读懂的配置错误.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed CUDA preflight failed: torch.cuda.is_available() is False, "
            "but multi-GPU inference was requested."
        )

    local_rank = _get_local_rank()
    visible_device_count = torch.cuda.device_count()
    if visible_device_count < expected_local_workers:
        raise RuntimeError(
            "Distributed CUDA preflight failed: "
            f"torchrun requested {expected_local_workers} local workers, "
            f"but this process only sees {visible_device_count} CUDA device(s). "
            f"LOCAL_RANK={local_rank}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
            "Please reduce --nproc-per-node / ulysses_degree / ring_degree, "
            "or expose more local GPUs before retrying."
        )
    if local_rank >= visible_device_count:
        raise RuntimeError(
            "Distributed CUDA preflight failed: "
            f"LOCAL_RANK={local_rank} is out of range for {visible_device_count} visible CUDA device(s). "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
        )
    return local_rank

def set_multi_gpus_devices(ulysses_degree, ring_degree, classifier_free_guidance_degree=1):
    if ulysses_degree > 1 or ring_degree > 1 or classifier_free_guidance_degree > 1:
        if get_sp_group is None:
            raise RuntimeError("xfuser is not installed.")
        expected_local_workers = _get_local_world_size(
            ring_degree * ulysses_degree * classifier_free_guidance_degree
        )
        local_rank = _validate_local_cuda_topology(expected_local_workers)
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        print('parallel inference enabled: ulysses_degree=%d ring_degree=%d classifier_free_guidance_degree=%d rank=%d world_size=%d' % (
            ulysses_degree, ring_degree, classifier_free_guidance_degree, dist.get_rank(),
            dist.get_world_size()))
        assert dist.get_world_size() == ring_degree * ulysses_degree * classifier_free_guidance_degree, \
                    "number of GPUs(%d) should be equal to ring_degree * ulysses_degree * classifier_free_guidance_degree." % dist.get_world_size()
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=ring_degree * ulysses_degree,
                classifier_free_guidance_degree=classifier_free_guidance_degree,
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree)
        # 这里显式绑定到 LOCAL_RANK 对应的 GPU.
        # 这样 FSDP / 后续 CUDA 张量分配都会落在当前进程真实可见的本地设备上.
        device = torch.device(f"cuda:{local_rank}")
        print('rank=%d device=%s' % (get_world_group().rank, str(device)))
    else:
        device = "cuda"
    return device

def sequence_parallel_chunk(x, dim=1):
    if get_sequence_parallel_world_size is None or not model_parallel_is_initialized():
        return x

    sp_world_size = get_sequence_parallel_world_size()
    if sp_world_size <= 1:
        return x

    sp_rank = get_sequence_parallel_rank()
    sp_group = get_sp_group()

    if x.size(1) % sp_world_size != 0:
        raise ValueError(f"Dim 1 of x ({x.size(1)}) not divisible by SP world size ({sp_world_size})")

    chunks = torch.chunk(x, sp_world_size, dim=1)
    x = chunks[sp_rank]

    return x

def sequence_parallel_all_gather(x, dim=1):
    if get_sequence_parallel_world_size is None or not model_parallel_is_initialized():
        return x

    sp_world_size = get_sequence_parallel_world_size()
    if sp_world_size <= 1:
        return x  # No gathering needed

    sp_group = get_sp_group()
    gathered_x = sp_group.all_gather(x, dim=dim)
    return gathered_x
