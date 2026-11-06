"""Untrusted worker entrypoint, run only inside rsi-isolate by the controller.

Nothing this process prints, measures or sends is accepted as evidence except
the literal acknowledgement {'kind': 'done', 'nonce': <controller nonce>}; the
controller keeps the clock, the snapshots, the checks and the result file.
"""
import argparse
import os
import socket
import time
from isolation.guard import install
from isolation.wire import receive, send

PROTOCOL = '0.3.0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    channel = socket.socket(fileno=args.fd)
    install()  # before torch, extension compilation, or candidate module import
    import torch
    import benchmark as b
    from isolation.vmm import decode_maps, import_allocations
    # The source policy is a development contract, not the authority boundary.
    # Scratch-cap baseline: a CUDA context on every device before any candidate
    # code exists in this process; the controller's trusted monitor records
    # this process tree's GPU memory now (isolation/gpumem.py).
    for i in range(torch.cuda.device_count()):
        torch.empty(1, device=torch.device('cuda', i))
    b.sync()
    send(channel, {'kind': 'context', 'protocol': PROTOCOL})
    message, fds = receive(channel)
    if message != {'kind': 'load'} or fds:
        raise ValueError('Expected load command')
    fn = b.load_candidate(args.candidate)
    profile = os.environ.get('KV_WORKER_PROFILE') == '1'
    send(channel, {'kind': 'ready', 'protocol': PROTOCOL})
    case = allocations = None
    while True:
        # Spin while idle so request wake-up latency is not charged to any implementation.
        message, fds = receive(channel, allow_fds=True, spin=86400)
        kind = message.get('kind')
        if kind == 'stop':
            break
        if kind == 'prepare':
            if case is not None:
                raise ValueError('Case already prepared')
            case, allocations = import_allocations(message, fds)
            send(channel, {'kind': 'prepared', 'nonce': message['nonce']})
        elif kind == 'release':
            case = None
            for allocation in reversed(allocations or []):
                allocation.close()
            allocations = None
            torch.cuda.empty_cache()
            send(channel, {'kind': 'released', 'nonce': message['nonce']})
        elif kind == 'transfer':
            if fds or case is None:
                raise ValueError('Transfer without a prepared case')
            t0 = time.perf_counter_ns()
            plan = dict(case)
            plan.update(decode_maps(message['maps'], case))
            exposed = b.candidate_view(plan)
            fn(exposed)
            b.sync()
            send(channel, {'kind': 'done', 'nonce': message['nonce']})
            if profile:
                # Untrusted development diagnostics; never parsed by the controller.
                print('WORKER_PROFILE request_to_ack_us=%.1f' % ((time.perf_counter_ns() - t0) / 1e3), flush=True)
            try:
                b.validate_candidate_view(exposed, plan)
            except AssertionError as exc:  # feedback only; the controller checks outputs itself
                print('WORKER_NOTE candidate mutated its plan view: %s' % exc, flush=True)
            del plan, exposed
        else:
            raise ValueError('Unexpected command')


if __name__ == '__main__':
    main()
