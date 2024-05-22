from typing_extensions import deprecated  # Python 3.13+

from .parallel_apply import parallel_apply
from .replicate import replicate
from .data_parallel import DataParallel, data_parallel
from .scatter_gather import gather, scatter
from .distributed import DistributedDataParallel

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel',
           'DataParallel', 'DistributedDataParallel']

@deprecated(
    "`torch.nn.parallel.DistributedDataParallelCPU` is deprecated, "
    "please use `torch.nn.parallel.DistributedDataParallel` instead.",
)
def DistributedDataParallelCPU(*args, **kwargs):
    return DistributedDataParallel(*args, **kwargs)
