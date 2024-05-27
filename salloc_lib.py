import enum
import os


class Partitions(enum.Enum):
    LONG = "long"
    MAIN = "main"
    UNKILLABLE = "unkillable"


def main(prep_command, num_gpus=1, num_nodes=1, *, unkillable=False, main=False):

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
