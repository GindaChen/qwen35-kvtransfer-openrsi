"""Worker messages cannot supply executable objects, oversized data or FDs."""
import json
import os
import socket
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'environment'))
from isolation.wire import MAX_MESSAGE, receive, send

class Wire(unittest.TestCase):
    def setUp(self): self.a,self.b=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
    def tearDown(self): self.a.close();self.b.close()
    def test_json(self):
        send(self.a,{'kind':'done','nonce':'example'})
        self.assertEqual(receive(self.b),({'kind':'done','nonce':'example'},[]))
    def test_no_worker_fds(self):
        fd=os.open('/dev/null',os.O_RDONLY)
        try:
            send(self.a,{'kind':'done'},[fd])
            with self.assertRaises(ValueError):receive(self.b)
        finally:os.close(fd)
    def test_truncated(self):
        self.a.send(b' '* (MAX_MESSAGE+1))
        with self.assertRaises(RuntimeError):receive(self.b)
    def test_non_object(self):
        self.a.send(b'[]')
        with self.assertRaises(ValueError):receive(self.b)
    def test_invalid_json(self):
        self.a.send(b'pickle is not a protocol')
        with self.assertRaises(json.JSONDecodeError):receive(self.b)
    def test_outgoing_bound(self):
        with self.assertRaises(ValueError):send(self.a,{'x':'x'*MAX_MESSAGE})

if __name__=='__main__': unittest.main()
