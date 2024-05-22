from typing_extensions import deprecated  # Python 3.13+

@deprecated(
    "Usage of `backward_compatibility.worker_init_fn` is deprecated "
    "as `DataLoader` automatically applies sharding in every worker",
)
def worker_init_fn(worker_id):
    pass
