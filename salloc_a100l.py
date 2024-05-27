#!/usr/bin/env python
import functools
import itertools
import logging

import fire
import rich
import rich.traceback

import salloc_lib


rich.traceback.install()
LOGGER = logging.getLogger()


def prep_command(executable, num_gpus, num_nodes, partition, extra_kwargs):
    LOGGER.debug("salloc_a100l.prep_command", locals())

    print(f"{partition = }")

    if partition == salloc_lib.Partitions.UNKILLABLE:
        num_gpus = 1
        mem = 32
        cpus = 5
    elif partition == salloc_lib.Partitions.MAIN:
        num_gpus = 2
        mem = 48
        cpus = 8
    else:
        assert partition == salloc_lib.Partitions.LONG, partition
        mem = num_gpus * 100
        cpus = num_gpus * 10

    assert num_gpus

    args = [
        executable,
        f"--gres=gpu:a100l:{num_gpus}",
        f"--mem={mem}GB",
        f"-c", f"{cpus}",
        f"-N", f"{num_nodes}",
        f"--partition", f"{partition.value}",
    ] + list(itertools.chain.from_iterable([f"--{k}", v] for k, v in extra_kwargs.items()))

    rich.print(args)

    return args


if __name__ == "__main__":
    fire.Fire(functools.partial(salloc_lib.main(prep_command)))
