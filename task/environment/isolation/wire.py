"""Bounded JSON messages and explicit file descriptors; never pickle worker data."""
import array
import json
import socket
import time

MAX_MESSAGE = 131072
MAX_FDS = 12


def send(sock, message, fds=()):
    data = json.dumps(message, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > MAX_MESSAGE or len(fds) > MAX_FDS:
        raise ValueError('IPC message exceeds limits')
    ancillary = []
    if fds:
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', fds))]
    if sock.sendmsg([data], ancillary) != len(data):
        raise RuntimeError('Short IPC message')


def receive(sock, *, allow_fds=False, spin=None):
    """Receive one message. ``spin`` (seconds) busy-polls instead of sleeping in
    the kernel, removing scheduler wake-up latency from timed round trips."""
    space = socket.CMSG_SPACE(MAX_FDS * array.array('i').itemsize)
    if spin is None:
        data, ancillary, flags, _ = sock.recvmsg(MAX_MESSAGE, space, socket.MSG_CMSG_CLOEXEC)
    else:
        deadline = time.monotonic() + spin
        while True:
            try:
                data, ancillary, flags, _ = sock.recvmsg(
                    MAX_MESSAGE, space, socket.MSG_CMSG_CLOEXEC | socket.MSG_DONTWAIT)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError('IPC receive timed out')
    received = []
    try:
        for level, kind, content in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                raise ValueError('Unexpected IPC control message')
            values = array.array('i')
            values.frombytes(content)
            received.extend(values)
        if not data or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError('Closed or truncated IPC channel')
        if received and not allow_fds:
            raise ValueError('Worker must not return file descriptors')
        message = json.loads(data)
        if type(message) is not dict:
            raise ValueError('IPC message must be an object')
        return message, received
    except BaseException:
        import os
        for fd in received:
            os.close(fd)
        raise
