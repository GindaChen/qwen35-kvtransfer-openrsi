"""Additional process confinement installed before importing candidate libraries.

rsi-isolate owns filesystem/IP-network confinement and descendant cleanup.
This inherited seccomp filter additionally rejects tracing, cross-process
memory access, namespace/mount escape, io_uring, bpf/perf, signals to any
other process (needed on kernels lacking Landlock signal scope) and reaching
any socket endpoint (connect, addressed sendto, AF_UNIX datagrams). The
controller makes itself non-dumpable too.
"""
import ctypes
import errno
import os
import platform


def protect_parent():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0):  # PR_SET_DUMPABLE = 0
        raise OSError(ctypes.get_errno(), 'Cannot protect Judge process memory')


def install():
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('Worker guard must be installed before creating threads')
    machine = platform.machine()
    if machine == 'aarch64':
        arch, clone = 0xC00000B7, 220
        signals = [129, 138, 131, 240]  # kill, rt_sigqueueinfo, tgkill, rt_tgsigqueueinfo
        denied = [130, 424, 117, 270, 271, 438, 440, 239, 97, 268, 40,
                  39, 41, 51, 428, 429, 430, 431, 432, 433, 442, 425,
                  280, 241, 282]
    elif machine == 'x86_64':
        arch, clone = 0xC000003E, 56
        signals = [62, 129, 234, 297]
        denied = [200, 424, 101, 310, 311, 438, 440, 279, 272, 308, 165,
                  166, 155, 161, 428, 429, 430, 431, 432, 433, 442, 425,
                  321, 298, 323]
    else:
        raise RuntimeError('Unsupported seccomp architecture')
    # setpgid/setsid stay allowed: build tools (ninja's posix_spawn) create
    # process groups, rsi-isolate ends every descendant as their subreaper, and
    # the signal rules below already confine signals to the worker itself.
    # Classic BPF: every syscall gets a fresh nr load after checking arguments.
    ins = [(0x20, 0, 0, 4), (0x15, 1, 0, arch), (0x06, 0, 0, 0x80000000),
           (0x20, 0, 0, 0)]
    if machine == 'x86_64':
        ins += [(0x45, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000)]
    deny = 0x00050000 | errno.EPERM
    for nr in denied:
        ins += [(0x15, 0, 1, nr), (0x06, 0, 0, deny)]
    # Returning ENOSYS for clone3 lets pthreads use inspectable clone instead.
    ins += [(0x15, 0, 1, 435), (0x06, 0, 0, 0x00050000 | errno.ENOSYS)]
    for nr in signals:
        ins += [(0x15, 0, 4, nr), (0x20, 0, 0, 16),
                (0x15, 1, 0, os.getpid()), (0x06, 0, 0, deny),
                (0x06, 0, 0, 0x7FFF0000)]
    # Socket containment independent of the kernel's Landlock ABI. The CUDA
    # driver must create (bind/listen) its own abstract AF_UNIX SEQPACKET
    # socket during cuInit, so socket creation cannot simply be refused on
    # kernels without Landlock abstract-socket scoping. Instead the worker may
    # never reach another endpoint: connect() is refused, sendto() with an
    # explicit address is refused, and AF_UNIX datagram sockets (which can
    # address peers through sendmsg's msg_name) cannot be created.
    if machine == 'aarch64':
        socket_nr, socketpair_nr, connect_nr, sendto_nr = 198, 199, 203, 206
    else:
        socket_nr, socketpair_nr, connect_nr, sendto_nr = 41, 53, 42, 44
    allow = 0x7FFF0000
    ins += [(0x15, 0, 1, connect_nr), (0x06, 0, 0, deny)]
    ins += [(0x15, 0, 6, sendto_nr),
            (0x20, 0, 0, 48), (0x15, 0, 3, 0),   # dest_addr low word
            (0x20, 0, 0, 52), (0x15, 0, 1, 0),   # dest_addr high word
            (0x06, 0, 0, allow), (0x06, 0, 0, deny)]
    for nr in (socket_nr, socketpair_nr):
        ins += [(0x15, 0, 7, nr),
                (0x20, 0, 0, 16), (0x15, 0, 3, 1),   # domain == AF_UNIX?
                (0x20, 0, 0, 24), (0x54, 0, 0, 0xF),  # type & SOCK_TYPE_MASK
                (0x15, 1, 0, 2),                      # SOCK_DGRAM -> deny
                (0x06, 0, 0, allow), (0x06, 0, 0, deny)]
    # Reject new namespaces, CLONE_PARENT and CLONE_UNTRACED. Ordinary process
    # creation and threads stay within rsi-isolate's descendant cleanup tree.
    ins += [(0x15, 0, 4, clone), (0x20, 0, 0, 16),
            (0x45, 0, 1, 0x7E828000), (0x06, 0, 0, deny),
            (0x06, 0, 0, 0x7FFF0000), (0x06, 0, 0, 0x7FFF0000)]
    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
    array = (Filter * len(ins))(*(Filter(*item) for item in ins))
    program = Program(len(ins), array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot set no_new_privs')
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot install worker seccomp filter')
