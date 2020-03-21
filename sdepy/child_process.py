from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    if rank != 0:
        print("rank", rank)
        sde = comm.recv(source=0, tag=MPI.ANY_TAG)
        print(sde.step())

        comm.send("hi", dest=0, tag=0)

    comm.Disconnect()
