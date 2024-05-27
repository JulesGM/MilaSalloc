#!/usr/bin/env python
import enum
import itertools
import logging
import os

import fire
import rich
import rich.traceback

import salloc_lib
import subprocess as sp


rich.traceback.install()
LOGGER = logging.getLogger()


class Partitions(enum.Enum):
    LONG = "long"
    MAIN = "main"
    UNKILLABLE = "unkillable"


def prep_command(executable, num_gpus, num_nodes, partition, extra_kwargs):
    LOGGER.debug("salloc_a100l.prep_command", locals())

    print(f"{partition = }")

    if partition == Partitions.UNKILLABLE:
        num_gpus = 1
        mem = 32
        cpus = 5
    elif partition == Partitions.MAIN:
        num_gpus = 2
        mem = 48
        cpus = 8
    else:
        assert partition == Partitions.LONG, partition
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


def main(num_gpus=1, num_nodes=1, *, unkillable=False, main=False):

    assert isinstance(unkillable, bool), type(unkillable)
    assert isinstance(main, bool), type(main)
    assert ((not unkillable) and (not main)) or (unkillable ^ main), f"{unkillable = }, {main = }"
    partition = Partitions.LONG
    if unkillable:
        partition = Partitions.UNKILLABLE
    elif main:
        partition = Partitions.MAIN

    executable = "salloc"
    command = prep_command(executable, num_gpus, num_nodes, partition, {})
    os.execvp(executable, command)


if __name__ == "__main__":
    # salloc_lib.start(prep_command)
    fire.Fire(main)
