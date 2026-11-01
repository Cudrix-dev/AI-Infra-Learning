"""Optimizer-only CUDA Graph experiment: no forward/backward/NCCL capture.

Both graph-ready eager and replay use identical AdamW settings and persistent
gradient tensors. The legacy harness defaults remain unchanged.
"""
import torch


def prepare_optimizer(model, cfg, device, mode, lr_lambda):
    if device.type != "cuda":
        raise ValueError("the optimizer graph experiment requires CUDA")
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=torch.tensor(cfg.lr, device=device), weight_decay=cfg.weight_decay,
        foreach=False, fused=False, capturable=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    for p in params:
        p.grad = torch.zeros_like(p)
    originals = [p.detach().clone() for p in params]
    # Warm allocator and AdamW's lazy state on a side stream. These are dummy
    # updates; restore parameters AND moments/counters before the first real step.
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(3):
            optimizer.step()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            optimizer.step()
    with torch.no_grad():
        for p, original in zip(params, originals):
            p.copy_(original)
            p.grad.zero_()
            for value in optimizer.state[p].values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
    torch.cuda.synchronize(device)
    del originals
    pointers = [p.grad.data_ptr() for p in params]
    lr_pointer = optimizer.param_groups[0]["lr"].data_ptr()

    def update():
        # A bucket-view rebuild would rebind .grad and invalidate this graph.
        if [p.grad.data_ptr() for p in params] != pointers:
            raise RuntimeError("gradient addresses changed after capture/preparation")
        if optimizer.param_groups[0]["lr"].data_ptr() != lr_pointer:
            raise RuntimeError("scheduler replaced the captured learning-rate tensor")
        if graph is None:
            optimizer.step()
        else:
            graph.replay()
        scheduler.step()  # tensor LR is updated in place, on the replay stream

    return optimizer, scheduler, update
