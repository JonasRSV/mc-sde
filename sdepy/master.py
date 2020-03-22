import sys
import os
from mpi4py import MPI
from sdepy import child
from sdepy.core import Job
import inspect


def run(job: Job, processes: int, parent: str = None):
    # Try to infer partent file if possible
    # It is needed so that the MPI jobs can import the functions
    if parent is None:
        frame = inspect.stack()[1]
        parent = frame[0].f_code.co_filename

    info = MPI.Info.Create()
    info.Set('env', f"PYTHONPATH={os.environ['PYTHONPATH']}:{os.path.dirname(parent)}")

    comm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args=[child.__file__],
        maxprocs=processes,
        info=info
    )

    comm.bcast(job, root=MPI.ROOT)
    pdfs = comm.gather(None, root=MPI.ROOT)

    # TODO merge with recursive doubling approach
    pdf = pdfs[0]
    if len(pdfs) > 0:
        for _pdf in pdfs[1:]:
            pdf = pdf.merge(_pdf)

    comm.Disconnect()

    return pdf
