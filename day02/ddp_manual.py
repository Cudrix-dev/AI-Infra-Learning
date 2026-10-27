import torch
import torch.distributed as dist


def sync_grad(params, world):
    """
    framework:
        if world==1:
            return
        else:
            for p in params:
                # 1. find the gradient of p
                grad = p.grad
                # 2. all-reduce (SUM): every rank gets the summed gradient
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                # 3. divide the result by world size to get the average gradient
                grad.div_(world)
    """


def broadcast_params(params, world):
    """
    Used to synchronize the model parameters across all ranks before training.
    framework:
        if world == 1:
            return
        else:
            for p in params:
                # 1. get the parameter data
                data = p.data
                # 2. broadcast the parameter data from rank 0 to all other ranks
                dist.broadcast(data, src=0)
    """


"""
Call Order:
1. broadcast_params(params, world)
2. for each training step:
    a. zero_grad
    b. forward pass
    c. compute loss
    d. backward pass
    e. sync_grad(params, world)
    f. optional gradient clipping
    g. optimizer step and scheduler step

Learning assumptions: every rank has the same parameters in the same order,
every parameter receives a gradient, and local losses average over equal
numbers of valid tokens. Before real communication, missing gradients must
be checked consistently across ranks. Parameter broadcast here does not
synchronize buffers; this model's causal mask is identical by construction.
"""


def ring_all_reduce(gradient, rank, world):
    """
    if world == 1:
        return gradient
    core step: scatter reduce and all gather
    W processes (W ranks), current rank index = r

    next_rank = (rank + 1) % world
    previous_rank = (rank - 1) % world
    0. flatten FIRST, then split into W chunks; unequal/empty chunks are allowed
    # Work on a copy so that the final write-back is explicit, even if the
    # original gradient is non-contiguous. The chunks are views into flat.
    flat = gradient.detach().reshape(-1).clone()
    chunks = list(torch.tensor_split(flat, world))

    1. scatter-reduce
    for step in range(world-1):
        send_idx=(rank-step)%world
        recv_idx=(rank-step-1)%world
        # Receive size follows recv_idx, NOT send_idx.
        recv_buffer = torch.empty_like(chunks[recv_idx])
        received = exchange(
            chunks[send_idx], recv_buffer, next_rank, previous_rank
        )
        chunks[recv_idx].add_(received)

    2. all-gather
    for step in range(world-1):
        send_idx=(rank+1-step)%world
        recv_idx=(rank-step)%world
        chunks[recv_idx] = exchange(
            chunks[send_idx], recv_buffer, next_rank, previous_rank
        )

    # In the executable version, run this under torch.no_grad().
    gradient.copy_(flat.view_as(gradient))
    return gradient  # SUM, same shape, original gradient updated in place

    Caller: replace dist.all_reduce(grad, SUM) with ring_all_reduce(...),
    then keep grad.div_(world). Do not average twice.


    """


def exchange(send_buffer, recv_buffer, send_to, recv_from):
    """
    send tensor to next_rank
    recv tensor from previous_rank
    return received tensor

    1. caller provides recv_buffer sized for the EXPECTED RECEIVED chunk
    # All ranks use the same split sizes. Sending chunk k matches receiving
    # chunk k on the next rank, even when some chunks contain zero elements.

    2. describe non-empty sends and receives independently
    ops = []
    if send_buffer.numel() > 0:
        ops.append(dist.P2POp(dist.isend, send_buffer, send_to))
    if recv_buffer.numel() > 0:
        ops.append(dist.P2POp(dist.irecv, recv_buffer, recv_from))

    3. submit the batch before waiting; an empty batch needs no communication
    requests = dist.batch_isend_irecv(ops) if ops else []

    4. wait for the operations to complete
    for request in requests:
        request.wait()

    return recv_buffer

    Runtime assumptions: process group initialized and current CUDA device
    set correctly. For NCCL, initialize the communicator with an all-rank
    collective (e.g. the initial parameter broadcast) before this helper;
    skipping empty chunks may otherwise omit ranks from the first P2P call.
    """
