from sdepy.core import SDE, Job, PDF
from mpi4py import MPI
import numpy as np


def _fit_and_gather(pdf: PDF, particles: np.ndarray, comm):
    pdf.fit(particles)
    comm.gather(pdf, root=0)


def raw_run(comm, job: Job, rank: int):
    sde = job.sde
    pdf = job.pdf

    particles = None
    while not sde.stop():
        _, particles = sde.step()

    _fit_and_gather(pdf, particles, comm)


def video_run(comm, job: Job, rank: int):
    sde = job.sde
    pdf = job.pdf
    spf = job.settings["steps_per_frame"] # steps per frame

    particles = None
    steps = 1
    while not sde.stop():
        _, particles = sde.step()
        if steps % spf == 0:
            _fit_and_gather(pdf, particles, comm)

        steps += 1

    #_fit_and_gather(pdf, particles, comm)


jobs = {
    Job.RAW: raw_run,
    Job.Video: video_run
}

if __name__ == "__main__":
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    job = None
    job = comm.bcast(job, root=0)

    job.init_on_process()

    jobs[job.mode](comm, job, rank)
    comm.Disconnect()
