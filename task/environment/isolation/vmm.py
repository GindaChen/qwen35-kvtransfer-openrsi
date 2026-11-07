"""Dedicated CUDA VMM allocations exported as POSIX FDs, never allocator slabs.

Each exported tensor is its own cuMemCreate allocation, so a worker mapping
reaches exactly that tensor. Controller snapshots, staging and captures use the
controller's ordinary allocator and are never exported. Allocations are
published once per case (untimed); payload freshness per invocation is
enforced by the controller's timed fill (see isolated_benchmark.py).
"""
import math
import os


def checked(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f'CUDA driver error {int(error)}')
    return values[0] if len(values) == 1 else tuple(values)


class Allocation:
    def __init__(self, device, shape, dtype, fd=None, size=None):
        import torch
        from cuda.bindings import driver as cu
        self.cu = cu
        self.device, self.shape, self.dtype = device, tuple(shape), dtype
        self.handle = self.address = None
        self.mapped = False
        self.size = 0
        self.tensor = None
        torch.cuda.set_device(device)
        checked(cu.cuInit(0))
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        granularity = checked(cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        needed = math.prod(shape) * (2 if dtype == 'bfloat16' else 4)
        self.size = math.ceil(needed / granularity) * granularity if size is None else size
        if self.size < needed or self.size % granularity:
            raise ValueError('Invalid VMM allocation size')
        try:
            if fd is None:
                self.handle = checked(cu.cuMemCreate(self.size, prop, 0))
            else:
                self.handle = checked(cu.cuMemImportFromShareableHandle(
                    fd, cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))
            self.address = checked(cu.cuMemAddressReserve(self.size, 0, 0, 0))
            checked(cu.cuMemMap(self.address, self.size, 0, self.handle, 0))
            self.mapped = True
            access = []
            for index in range(4):
                desc = cu.CUmemAccessDesc()
                desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                desc.location.id = index
                desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                access.append(desc)
            checked(cu.cuMemSetAccess(self.address, self.size, access, len(access)))
            # BF16 is represented as raw int16 in the array interface, then viewed
            # without copying. The owning Allocation must outlive this tensor.
            interface = type('DeviceArray', (), {})()
            interface.__cuda_array_interface__ = {
                'shape': self.shape, 'strides': None,
                'typestr': '<i2' if dtype == 'bfloat16' else '<f4',
                'data': (int(self.address), False), 'version': 3,
            }
            self.tensor = torch.as_tensor(interface, device=f'cuda:{device}')
            if dtype == 'bfloat16':
                self.tensor = self.tensor.view(torch.bfloat16)
        except BaseException:
            self.close()
            raise

    def export(self):
        return checked(self.cu.cuMemExportToShareableHandle(
            self.handle,
            self.cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0))

    def descriptor(self):
        return dict(device=self.device, shape=self.shape, dtype=self.dtype, size=self.size)

    def close(self):
        self.tensor = None
        if self.address is not None:
            if self.mapped:
                checked(self.cu.cuMemUnmap(self.address, self.size))
                self.mapped = False
            checked(self.cu.cuMemAddressFree(self.address, self.size))
            self.address = None
        if self.handle is not None:
            checked(self.cu.cuMemRelease(self.handle))
            self.handle = None


def allocate_plan(p, d, bs, ctx, layout, seed):
    import benchmark as b
    plan = b.maps(bs, ctx, layout, seed)
    plan.update(p=p, d=d, bs=bs, ctx=ctx, src=[], dst=[])
    allocations = []
    try:
        for group, tp, offset in [('src', p, 0), ('dst', d, p)]:
            for rank in range(tp):
                values = {}
                shapes = {
                    'kv': ((10, plan['cap'], 2 // tp, b.BLOCK, 512), 'bfloat16'),
                    'conv': ((30, plan['slotcap'], 3, 8192 // tp), 'bfloat16'),
                    'recurrent': ((30, plan['slotcap'], 32 // tp, 128, 128), 'float32'),
                }
                for name, (shape, dtype) in shapes.items():
                    allocation = Allocation(offset + rank, shape, dtype)
                    allocations.append(allocation)
                    values[name] = allocation.tensor
                plan[group].append(values)
        return plan, allocations
    except BaseException:
        for allocation in reversed(allocations):
            allocation.close()
        raise


MAP_KEYS = ('src_pages', 'dst_pages', 'src_slots', 'dst_slots')
SCALARS = ('p', 'd', 'bs', 'ctx', 'cap', 'slotcap')


def describe_allocations(plan, allocations, nonce):
    """Per-case publication: tensor shapes and VMM handles, no maps or payload."""
    return {
        'kind': 'prepare', 'nonce': nonce,
        'scalars': {k: plan[k] for k in SCALARS},
        'allocations': [allocation.descriptor() for allocation in allocations],
    }


def encode_maps(plan):
    """Per-invocation request: little-endian int64 maps, base64 in JSON."""
    import base64
    return {k: base64.b64encode(plan[k].to(dtype=__import__('torch').int64).contiguous().numpy().tobytes()).decode()
            for k in MAP_KEYS}


def decode_maps(maps, scalars):
    import base64
    import torch
    if type(maps) is not dict or set(maps) != set(MAP_KEYS):
        raise ValueError('Invalid map message')
    out = {}
    for key in MAP_KEYS:
        raw = base64.b64decode(maps[key], validate=True)
        if len(raw) % 8:
            raise ValueError('Invalid map payload')
        out[key] = torch.frombuffer(bytearray(raw), dtype=torch.int64)
    n = scalars['bs'] * -(-scalars['ctx'] // 128)
    if out['src_pages'].numel() != n or out['dst_pages'].numel() != n or \
            out['src_slots'].numel() != scalars['bs'] or out['dst_slots'].numel() != scalars['bs']:
        raise ValueError('Map length does not match the case')
    return out


def import_allocations(message, fds):
    plan = dict(message['scalars'])
    allocations = []
    if len(fds) != len(message['allocations']):
        raise ValueError('Wrong VMM descriptor count')
    try:
        for desc, fd in zip(message['allocations'], fds):
            allocations.append(Allocation(fd=fd, **desc))
        offset = 0
        for group, tp in [('src', plan['p']), ('dst', plan['d'])]:
            plan[group] = []
            for rank in range(tp):
                plan[group].append(dict(zip(('kv', 'conv', 'recurrent'),
                    [a.tensor for a in allocations[offset:offset + 3]])))
                offset += 3
        return plan, allocations
    except BaseException:
        for allocation in reversed(allocations):
            allocation.close()
        raise
    finally:
        for fd in fds:
            os.close(fd)
