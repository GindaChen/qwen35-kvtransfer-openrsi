"""The native worker cannot signal/trace its parent, create namespaces or reach a socket endpoint."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'environment'

class Guard(unittest.TestCase):
    def test_native_syscalls(self):
        program=r'''
import ctypes,errno,json,os,platform
from isolation.guard import install
install()
libc=ctypes.CDLL(None,use_errno=True)
results={}
try:os.kill(os.getppid(),0)
except OSError as e:results['parent_signal']=e.errno
else:results['parent_signal']=0
machine=platform.machine()
for name,nr in [('unshare',97 if machine=='aarch64' else 272),('process_vm_readv',270 if machine=='aarch64' else 310),('clone3',435)]:
 ctypes.set_errno(0)
 result=libc.syscall(nr,0,0,0,0,0,0)
 results[name]=ctypes.get_errno() if result==-1 else 0
# Child processes inherit confinement. This must not signal the real parent.
pid=os.fork()
if pid==0:
 try:os.kill(os.getppid()-1,0)
 except OSError as e:os._exit(0 if e.errno==errno.EPERM else 2)
 os._exit(3)
_,status=os.waitpid(pid,0)
results['child_inherits']=os.waitstatus_to_exitcode(status)==0
import socket
listener=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)  # CUDA's own uvmfd listener shape
listener.bind('\0kv-guard-test-%d'%os.getpid());listener.listen(1)
results['unix_listen']=0
probe=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:probe.connect('\0kv-guard-test-%d'%os.getpid())
except OSError as e:results['connect']=e.errno
else:results['connect']=0
try:probe.sendto(b'x','\0kv-guard-test-%d'%os.getpid())
except OSError as e:results['sendto_address']=e.errno
else:results['sendto_address']=0
for label,family,kind in [('unix_datagram',socket.AF_UNIX,socket.SOCK_DGRAM)]:
 try:socket.socket(family,kind)
 except OSError as e:results[label]=e.errno
 else:results[label]=0
try:socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
except OSError as e:results['datagram_pair']=e.errno
else:results['datagram_pair']=0
a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET);a.send(b'ok');results['pair_send']=b.recv(2)==b'ok'
pid=os.fork()
if pid==0:
 os.setpgid(0,0);os.setsid() if False else None;os._exit(0)
_,status=os.waitpid(pid,0)
results['setpgid']=os.waitstatus_to_exitcode(status)==0
print(json.dumps(results))
'''
        env=dict(os.environ,PYTHONPATH=str(ROOT))
        result=subprocess.run([sys.executable,'-c',program],env=env,capture_output=True,text=True,timeout=20,check=True)
        data=json.loads(result.stdout)
        self.assertEqual(data,dict(parent_signal=1,unshare=1,process_vm_readv=1,clone3=38,child_inherits=True,
                                unix_listen=0,connect=1,sendto_address=1,unix_datagram=1,datagram_pair=1,
                                pair_send=True,setpgid=True))

if __name__=='__main__':unittest.main()
