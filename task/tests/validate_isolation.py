"""Adversarial GB200 integration checks for the actual candidate boundary.

The generated hostile sources and native worker logs stay in the verifier's
private directory. Only the bounded aggregate report is suitable for publication.
"""
import argparse
import ctypes
import time
import inspect
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'environment'))
import torch
import benchmark as b
from isolated_benchmark import Worker, run_case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--isolator')
    parser.add_argument('--allow-hardware',action='store_true',
                        help='Development only: run on non-GB200 GPUs; the report says so')
    args=parser.parse_args()
    began=time.perf_counter()
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names)!=4 or not (args.allow_hardware or all('GB200' in n for n in names)):
        raise RuntimeError('Run isolation integration tests only on four delegated GB200 GPUs')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    # Hostile candidate sources must be readable by the isolated worker, and
    # rsi-isolate never grants anything under /logs/verifier (where `root`
    # lives on the Judge), so they go in a separate world-readable directory.
    # They hold no secret; the private canaries below stay under `root`.
    import tempfile
    fixtures=Path(tempfile.mkdtemp(prefix='kv-preflight-candidates-'));fixtures.chmod(0o755)
    private=root/'private';private.mkdir()
    secret=private/'canary.txt';secret.write_text('unchanged-private-canary')
    forged=private/'reward.json'
    memory=ctypes.c_uint64(0x3141592653589793)
    # Controller-owned listeners the worker must not reach (abstract + path).
    import socket as _socket
    abstract='\0kv-boundary-probe-'+os.urandom(8).hex()
    import tempfile
    listen_path=Path(tempfile.mkdtemp(prefix='kvp-'))/'c.sock'  # AF_UNIX paths are short
    listeners=[]
    for address in (abstract,str(listen_path)):
        sock=_socket.socket(_socket.AF_UNIX,_socket.SOCK_STREAM);sock.bind(address);sock.listen(4);sock.setblocking(False)
        listeners.append(sock)
    reference='import torch\n'+inspect.getsource(b.reference)+'\n'
    reports=[]

    def run(name,source,valid,**extra):
        t0=time.perf_counter()
        path=fixtures/(name+'.py');path.write_text(source)
        # Attack sources must really pass the current policy; a policy rejection
        # is not evidence that native code is contained by the OS boundary.
        b.validate_candidate_source(source)
        worker=Worker(path,root/(name+'.worker.log'),args.isolator)
        try:
            if valid:
                row=run_case(worker,1,1,1,129,'fragmented',0,1,12345)
                reports.append(dict(name=name,status='passed',median_us=row['median_us'],
                                    scratch_peak_gib=[round(v/2**30,3) for v in row['scratch_peak_bytes']],
                                    seconds=round(time.perf_counter()-t0,1)))
                return worker,row
            try:
                run_case(worker,1,1,1,129,'fragmented',0,1,12345)
            except (AssertionError,RuntimeError) as exc:
                entry=dict(name=name,status='rejected',seconds=round(time.perf_counter()-t0,1))
                if extra.get('expect'):
                    if extra['expect'] not in str(exc):
                        raise AssertionError(f'{name} rejected for the wrong reason: {exc}')
                    entry['reason']=extra['expect']
                reports.append(entry)
            else:raise AssertionError('Invalid candidate accepted: '+name)
        except BaseException:
            worker.close()
            raise
        finally:
            if not valid:worker.close()

    positive,row=run('reference',reference+'transfer=reference\n',True)
    positive.close()
    override='''import torch
registries=[]
for dispatch in ['CPU','CUDA']:
    registry=torch.library.Library('aten','IMPL',dispatch)
    registry.impl('equal',lambda a,b:True)
    registries.append(registry)
def transfer(plan):
    pass
'''
    run('dispatch_override',override,False)
    run('empty_maps',"import torch\ndef transfer(plan):\n for key in ['src_pages','dst_pages','src_slots','dst_slots']: plan[key]=torch.empty(0,dtype=torch.int64)\n",False)
    for family in ['kv','conv','recurrent']:
        source=reference+f"def transfer(plan):\n reference(plan)\n plan['dst'][0]['{family}'].zero_()\n"
        run('omitted_'+family,source,False)
    run('source_write',reference+"def transfer(plan):\n reference(plan)\n plan['src'][0]['conv'].zero_()\n",False)
    run('stale_output',reference+"calls=0\ndef transfer(plan):\n global calls\n calls+=1\n if calls==1: reference(plan)\n",False)

    # Scratch cap (2 GiB/GPU beyond the fixed pools), enforced by the trusted
    # NVML monitor against native allocations the Python policy never sees.
    def native_alloc(name,gib):
        code=f'''
import builtins,ctypes,torch
cuda=ctypes.CDLL('libcuda.so.1')
keep=[]
for dev in range(torch.cuda.device_count()):
 torch.cuda.set_device(dev);torch.empty(1,device=dev)
 ptr=ctypes.c_uint64()
 size=ctypes.c_size_t(int({gib}*2**30))
 if cuda.cuMemAlloc_v2(ctypes.byref(ptr),size):raise RuntimeError('cuMemAlloc failed')
 if cuda.cuMemsetD8_v2(ptr,ctypes.c_ubyte(1),size):raise RuntimeError('cuMemsetD8 failed')
 keep.append(ptr.value)
torch.cuda.set_device(0)
builtins.kv_scratch_probe=keep
'''
        cpp='#include <torch/extension.h>\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n PyRun_SimpleStringFlags('+json.dumps(code)+',nullptr);\n}\n'
        src=reference+'from torch.utils.cpp_extension import load_inline\n'
        src+=f'_native=load_inline(name="kv_scratch_{name}",cpp_sources='+repr(cpp)+',with_cuda=False,extra_cflags=["-O0"],verbose=False)\n'
        return src+'transfer=reference\n'
    run('scratch_within_cap',native_alloc('within',1.5),True)[0].close()
    run('scratch_over_cap',native_alloc('over',2.25),False,expect='scratch cap exceeded')
    # A child process the candidate starts is part of the worker tree.
    child='''
import os,sys,time,torch
x=torch.empty(int(2.5*2**30),dtype=torch.uint8,device='cuda:1');x.fill_(1);torch.cuda.synchronize(1)
open(sys.argv[1],'w').write('ready')
time.sleep(3600)
'''
    code=f'''
import builtins,os,subprocess,sys,tempfile,time
flag=os.path.join(tempfile.mkdtemp(),'ready')
builtins.kv_scratch_child=subprocess.Popen([sys.executable,'-c',{child!r},flag])
end=time.time()+300
while not os.path.exists(flag):
 if time.time()>end or builtins.kv_scratch_child.poll() is not None:raise RuntimeError('child allocation failed')
 time.sleep(0.2)
'''
    cpp='#include <torch/extension.h>\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n PyRun_SimpleStringFlags('+json.dumps(code)+',nullptr);\n}\n'
    src=reference+'from torch.utils.cpp_extension import load_inline\n'
    src+='_native=load_inline(name="kv_scratch_child",cpp_sources='+repr(cpp)+',with_cuda=False,extra_cflags=["-O0"],verbose=False)\n'
    run('scratch_child_process',src+'transfer=reference\n',False,expect='scratch cap exceeded')

    # Execute host Python through a permitted native extension, deliberately
    # bypassing Python AST restrictions. This must not alter Judge authority.
    payload=f'''
import ctypes,json,os,signal,socket,time,torch,benchmark
report={{}}
for name,path,mode in [('read_private',{str(secret)!r},'r'),('write_reward',{str(forged)!r},'w')]:
 try:
  with open(path,mode) as stream:
   stream.read() if mode=='r' else stream.write('{{"reward":9999}}')
 except OSError as exc: report[name]=exc.errno
 else: report[name]=0
try: os.kill({os.getpid()},signal.SIGCONT)
except OSError as exc: report['parent_signal']=exc.errno
else: report['parent_signal']=0
for name,path,mode in [('write_evaluator_code',{str(Path(b.__file__).resolve())!r},'a'),('controller_mem',{f'/proc/{os.getpid()}/mem'!r},'rb'),('controller_environ',{f'/proc/{os.getpid()}/environ'!r},'rb')]:
 try:
  with open(path,mode) as stream:
   stream.read(1) if 'r' in mode else stream.write('')
 except OSError as exc: report[name]=exc.errno
 else: report[name]=0
for label,family,kind in [('internet_socket',socket.AF_INET,socket.SOCK_STREAM),('unix_datagram',socket.AF_UNIX,socket.SOCK_DGRAM)]:
 try: sock=socket.socket(family,kind)
 except OSError as exc: report[label]=exc.errno
 else: sock.close();report[label]=0
for label,address in [('unix_connect_abstract',{abstract!r}),('unix_connect_path',{str(listen_path)!r})]:
 sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
 try: sock.connect(address)
 except OSError as exc: report[label]=exc.errno
 else: report[label]=0
 sock.close()
class IOV(ctypes.Structure):
 _fields_=[('base',ctypes.c_void_p),('size',ctypes.c_size_t)]
value=ctypes.c_uint64(0)
local=IOV(ctypes.addressof(value),8)
remote=IOV({ctypes.addressof(memory)},8)
libc=ctypes.CDLL(None,use_errno=True)
for name in ['process_vm_readv','process_vm_writev']:
 ctypes.set_errno(0)
 result=getattr(libc,name)({os.getpid()},ctypes.byref(local),1,ctypes.byref(remote),1,0)
 report[name]=ctypes.get_errno() if result==-1 else 0
print('NATIVE_BOUNDARY_REPORT '+json.dumps(report),flush=True)
benchmark.check=lambda *a:None
benchmark.check_source=lambda *a:None
benchmark.validate_candidate_view=lambda *a:None
benchmark.sync=lambda:None
torch.equal=lambda *a:True
time.perf_counter_ns=lambda:0
'''
    cpp='#include <torch/extension.h>\n#include <unistd.h>\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n PyRun_SimpleStringFlags('+json.dumps(payload)+',nullptr);\n m.def("wait", [](){usleep(40000);});\n}\n'
    native=reference+'from torch.utils.cpp_extension import load_inline\n'
    native+='_native=load_inline(name="kv_boundary_native_probe",cpp_sources='+repr(cpp)+',with_cuda=False,extra_cflags=["-O0"],verbose=False)\n'
    native+='calls=0\ndef transfer(plan):\n global calls\n calls+=1\n if calls<=3:\n  reference(plan)\n  _native.wait()\n'
    worker,row=run('native_authority',native,True)
    try:
        if row['median_us']<40000:raise AssertionError('Worker clock forged trusted elapsed time')
        try:run_case(worker,1,1,1,129,'fragmented',0,1,45678)
        except AssertionError:reports.append(dict(name='native_checker_and_early_ack',status='rejected'))
        else:raise AssertionError('Native checker monkeypatch bypassed trusted checker')
    finally:worker.close()
    # Early acknowledgement: queue a long GPU sleep, then the real transfer, and
    # acknowledge without synchronising (worker-side sync disabled natively).
    # The trusted capture must either time the late work or reject the result.
    sleep_cycles=200_000_000
    early=f'''
import benchmark,torch
benchmark.sync=lambda:None
torch.cuda.kv_probe_sleep=torch.cuda._sleep
'''
    cpp2='#include <torch/extension.h>\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n PyRun_SimpleStringFlags('+json.dumps(early)+',nullptr);\n}\n'
    early_src=reference+'from torch.utils.cpp_extension import load_inline\n'
    early_src+='_native=load_inline(name="kv_boundary_early_ack",cpp_sources='+repr(cpp2)+',with_cuda=False,extra_cflags=["-O0"],verbose=False)\n'
    early_src+='def transfer(plan):\n for dev in range(4):\n  with torch.cuda.device(dev):\n   torch.cuda.kv_probe_sleep(%d)\n reference(plan)\n' % sleep_cycles
    path=fixtures/'early_ack.py';path.write_text(early_src)
    b.validate_candidate_source(early_src)
    worker=Worker(path,root/'early_ack.worker.log',args.isolator)
    try:
        start=time.perf_counter()
        torch.cuda._sleep(sleep_cycles);torch.cuda.synchronize(0)
        sleep_us=(time.perf_counter()-start)*1e6
        try:row=run_case(worker,1,1,1,129,'fragmented',0,3,24680)
        except AssertionError:reports.append(dict(name='early_ack_unsynchronised',status='rejected'))
        else:
            if row['median_us']<0.8*sleep_us:raise AssertionError('Early acknowledgement hid late GPU work from the trusted clock')
            reports.append(dict(name='early_ack_unsynchronised',status='timed',median_us=row['median_us'],sleep_us=sleep_us))
    finally:worker.close()
    lines=(root/'native_authority.worker.log').read_text().splitlines()
    records=[json.loads(line.split('NATIVE_BOUNDARY_REPORT ',1)[1]) for line in lines if line.startswith('NATIVE_BOUNDARY_REPORT ')]
    if len(records)!=1:raise AssertionError('Native syscall probe did not execute')
    report=records[0]
    keys=['read_private','write_reward','write_evaluator_code','controller_mem','controller_environ','parent_signal',
          'internet_socket','unix_datagram','unix_connect_abstract','unix_connect_path','process_vm_readv','process_vm_writev']
    if not all(report.get(k) in (1,13) for k in keys):
        raise AssertionError('Native boundary escape: '+repr(report))
    for sock in listeners:
        try:sock.accept()
        except BlockingIOError:pass
        else:raise AssertionError('Worker reached a controller-owned socket')
        sock.close()
    if secret.read_text()!='unchanged-private-canary' or forged.exists() or memory.value!=0x3141592653589793:
        raise AssertionError('Private evaluator authority changed')
    result=dict(protocol='0.3.0',status='passed',cases=reports,native_syscall_denials=report,
                private_canaries_unchanged=True,worker_scores_used=False,hardware=names,
                scoring_hardware=all('GB200' in n for n in names),
                wall_seconds=round(time.perf_counter()-began,1))
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
