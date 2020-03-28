import sys
import os
from mpi4py import MPI
from sdepy import child
from sdepy.core import Job, Result, PDF
import inspect


def _gather_pdf(comm) -> PDF:
    # TODO merge with recursive doubling approach
    pdfs = comm.gather(None, root=MPI.ROOT)
    pdf = pdfs[0]
    if len(pdfs) > 0:
        for _pdf in pdfs[1:]:
            pdf = pdf.merge(_pdf)

    return pdf


def raw_collect(job, comm):
    # Distribute jobs
    comm.bcast(job, root=MPI.ROOT)
    pdf = _gather_pdf(comm)
    return Result([pdf])


def video_collect(job, comm):
    spf = job.settings["steps_per_frame"]  # steps per frame
    steps_left = job.sde.steps
    num_times_to_gather = (steps_left // spf)   # + 1 for final gather..
    # if deadlock occurs might be a off-by-one error here

    # Distribute jobs
    comm.bcast(job, root=MPI.ROOT)

    result = []
    for _ in range(num_times_to_gather):
        result.append(_gather_pdf(comm))

    return Result(result)


runners = {
    Job.RAW: raw_collect,
    Job.Video: video_collect
}


def run(job: Job, processes: int, parent: str = None) -> Result:
    # Try to infer parent file if possible
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

    result = runners[job.mode](job, comm)
    comm.Disconnect()

    return result
