from sdepy.core import SDE, Job
from mpi4py import MPI


def raw_run(comm, job: Job, rank: int):
    sde = job.sde
    pdf = job.pdf

    particles = None
    while not sde.stop():
        _, particles = sde.step()

    pdf.fit(particles)
    comm.gather(pdf, root=0)


jobs = {
    Job.RAW: raw_run
}

if __name__ == "__main__":
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    job = None
    job = comm.bcast(job, root=0)

    job.init_on_process()

    jobs[job.mode](comm, job, rank)
    comm.Disconnect()
