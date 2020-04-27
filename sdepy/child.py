from sdepy.core import SDE, Job, PDF
from mpi4py import MPI
import numpy as np
import time


def _recv_merges(pdf: PDF,
                 rank: int,
                 size: int,
                 doubling_participants: int):
    for sender in range(doubling_participants, size):
        receiver = (sender % doubling_participants)

        if rank == sender:
            MPI.COMM_WORLD.send(pdf, dest=receiver, tag=5)

        if rank == receiver:
            recv_pdf = MPI.COMM_WORLD.recv(source=sender, tag=MPI.ANY_TAG)
            pdf = pdf.merge(recv_pdf)

    return pdf


def _send_merges(pdf: PDF,
                 rank: int,
                 size: int,
                 doubling_participants: int):
    for receiver in range(doubling_participants, size):
        sender = (receiver % doubling_participants)

        if rank == sender:
            MPI.COMM_WORLD.send(pdf, dest=receiver, tag=5)

        if rank == receiver:
            recv_pdf = MPI.COMM_WORLD.recv(source=sender, tag=MPI.ANY_TAG)
            pdf = pdf.merge(recv_pdf)

    return pdf


def _recursive_doubling(pdf: PDF, rank: int, stages: int):
    for stage in range(1, stages + 1):
        base = int(np.float_power(2, stage))
        send_offset = int(np.float_power(2, stage - 1))

        local_base = rank - (rank % base)

        send_to = local_base + (rank + send_offset) % base
        receive_from = local_base + (rank - send_offset) % base

        #print("rank", rank, "pdf", pdf)
        MPI.COMM_WORLD.isend(pdf, dest=send_to, tag=0)
        recv_pdf = MPI.COMM_WORLD.recv(source=receive_from, tag=MPI.ANY_TAG)
        #print("recieved", rank, "from", receive_from)

        #print(f"{rank} merging with {receive_from}")

        pdf = pdf.merge(recv_pdf)
    return pdf


def _reduce_pdf(pdf: PDF, rank: int):
    size = comm.Get_size()

    stages = int(np.log2(size))
    doubling_participants = int(np.float_power(2, stages))
    pdf = _recv_merges(pdf=pdf,
                       rank=rank,
                       size=size,
                       doubling_participants=doubling_participants)

    if rank < doubling_participants:
        #print("doubling participant", rank)
        pdf = _recursive_doubling(pdf, rank, stages)

    #print("Recursive doubling done", rank)
    pdf = _send_merges(pdf=pdf,
                       rank=rank,
                       size=size,
                       doubling_participants=doubling_participants)

    return pdf


def _fit_and_gather(pdf: PDF, particles: np.ndarray, comm, rank: int):
    timestamp = time.time()
    #print(f"rank {rank} timestamp", int(timestamp) % 100)
    pdf.fit(particles)
    #print(f"Rank {rank} -- fit-time: {time.time() - timestamp}", int(time.time()) % 100)
    pdf = _reduce_pdf(pdf, rank)
    #print(f"Rank {rank} -- fit-double-time: {time.time() - timestamp}", int(time.time()) % 100)
    comm.gather(pdf, root=0)


def raw_run(comm, job: Job, rank: int):
    sde = job.sde
    pdf = job.pdf

    timestamp = time.time()
    particles = None
    while not sde.stop():
        _, particles = sde.step()

    print(f"Rank {rank} -- run-time: {time.time() - timestamp}")

    _fit_and_gather(pdf, particles, comm, rank)


def video_run(comm, job: Job, rank: int):
    sde = job.sde
    pdf = job.pdf
    spf = job.settings["steps_per_frame"]  # steps per frame

    steps = 1
    while not sde.stop():
        _, particles = sde.step()
        if steps % spf == 0:
            _fit_and_gather(pdf, particles, comm, rank)

        steps += 1


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
