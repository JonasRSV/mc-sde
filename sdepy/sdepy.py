import sys
from core import SDE
from mpi4py import MPI
import child_process


def run(sde: SDE, processes: int):
    comm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args=[child_process.__file__],
        maxprocs=processes,
        root=0
    )
    rank = comm.Get_rank()

    print("Rank master", rank)

    for target in range(1, processes):
        print("target", target)
        comm.send(sde, dest=target, tag=0)

    print("rank after", rank)
    print("Disconnect master")

    print("Rescv", comm.recv(source=1, tag=MPI.ANY_TAG))

    comm.Disconnect()
