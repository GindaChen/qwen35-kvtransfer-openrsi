"""Transfer-only Qwen3.5 state benchmark. Evaluator authority; do not edit during research."""
import argparse,importlib.util,json,math,statistics,time,subprocess,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
LENGTHS=[4096,8192,32768,65536,4097,10034,32749,65521]
BLOCK=128

def sync():
 for i in range(4):torch.cuda.synchronize(i)

def reference(plan):
 """Naive Torch: gather, peer-copy, scatter each overlapping rank/head range.
 Q/K/V convolution segments must be sliced separately for heterogeneous TP.
 """
 src,dst=plan['src'],plan['dst']; p,d=plan['p'],plan['d']
 sm,dm,ss,ds=[plan[k] for k in ['src_pages','dst_pages','src_slots','dst_slots']]
 for dr in range(d):
  target=dst[dr]; dg=p+dr
  di=dm.to(f'cuda:{dg}'); dsi=ds.to(f'cuda:{dg}')
  for sr in range(p):
   origin=src[sr]; si=sm.to(f'cuda:{sr}'); ssi=ss.to(f'cuda:{sr}')
   # Full attention: global KV head overlap, 2 heads total.
   lo=max(sr*2//p,dr*2//d); hi=min((sr+1)*2//p,(dr+1)*2//d)
   if lo<hi:
    sl=slice(lo-sr*2//p,hi-sr*2//p); dl=slice(lo-dr*2//d,hi-dr*2//d)
    for layer in range(10):
     packed=origin['kv'][layer,:,sl].index_select(0,si)
     remote=packed.to(f'cuda:{dg}')
     target['kv'][layer,:,dl].index_copy_(0,di,remote)
   # Fixed recurrent state: 32 global value heads.
   lo=max(sr*32//p,dr*32//d); hi=min((sr+1)*32//p,(dr+1)*32//d)
   if lo<hi:
    sl=slice(lo-sr*32//p,hi-sr*32//p); dl=slice(lo-dr*32//d,hi-dr*32//d)
    packed=origin['recurrent'][:,:,sl].index_select(1,ssi)
    target['recurrent'][:,:,dl].index_copy_(1,dsi,packed.to(f'cuda:{dg}'))
   # Three independently TP-sharded channel segments, not one contiguous shard.
   for seg,total in enumerate([2048,2048,4096]):
    so=sum([2048,2048,4096][:seg])//p; do=sum([2048,2048,4096][:seg])//d
    lo=max(sr*total//p,dr*total//d); hi=min((sr+1)*total//p,(dr+1)*total//d)
    if lo>=hi:continue
    sl=slice(so+lo-sr*total//p,so+hi-sr*total//p)
    dl=slice(do+lo-dr*total//d,do+hi-dr*total//d)
    packed=origin['conv'][:,:,:,sl].index_select(1,ssi)
    target['conv'][:,:,:,dl].index_copy_(1,dsi,packed.to(f'cuda:{dg}'))

def maps(bs,ctx,fragmented,seed):
 g=torch.Generator().manual_seed(seed); n=bs*math.ceil(ctx/BLOCK)
 cap=math.ceil(n*1.5)+17; slots=2*bs+7
 if fragmented:
  a=torch.randperm(cap,generator=g)[:n]; b=torch.randperm(cap,generator=g)[:n]
 else:a=torch.arange(n)+3; b=torch.arange(n)+9
 return dict(src_pages=a,dst_pages=b,src_slots=torch.randperm(slots,generator=g)[:bs],dst_slots=torch.randperm(slots,generator=g)[:bs],cap=cap,slotcap=slots)

def allocate(p,d,bs,ctx,fragmented,seed):
 plan=maps(bs,ctx,fragmented,seed); plan.update(p=p,d=d,bs=bs,ctx=ctx,src=[],dst=[])
 for group,tp,offset in [('src',p,0),('dst',d,p)]:
  for rank in range(tp):
   dev=f'cuda:{offset+rank}'
   plan[group].append({'kv':torch.empty((10,plan['cap'],2//tp,BLOCK,512),dtype=torch.bfloat16,device=dev),
    'conv':torch.empty((30,plan['slotcap'],3,8192//tp),dtype=torch.bfloat16,device=dev),
    'recurrent':torch.empty((30,plan['slotcap'],32//tp,128,128),dtype=torch.float32,device=dev)})
 return plan

def fill(plan,generation):
 # Integer patterns are exactly representable in both dtypes; each element varies.
 # Add rank and generation offsets to detect stale output and incorrect head owners.
 for r,src in enumerate(plan['src']):
  torch.cuda.set_device(r)
  for name,t in src.items():
   t.random_(0,64); t.add_((generation*7+r*17)%64)
 for target in plan['dst']:
  for t in target.values():t.fill_(-777)

def check(plan, snapshot):
 """Independent checker maps global heads and each convolution segment.
 Compare selected pages/slots and require all holes to preserve poison.
 """
 p,d=plan['p'],plan['d']; sm=plan['src_pages']; dm=plan['dst_pages'];ss=plan['src_slots'];ds=plan['dst_slots']
 for dr,target in enumerate(plan['dst']):
  dev=target['kv'].device; di=dm.to(dev); dsi=ds.to(dev)
  for name,count in [('kv',2),('recurrent',32)]:
   axis=2
   for local in range(count//d):
    glob=dr*(count//d)+local;sr=glob//(count//p);sh=glob%(count//p)
    source=snapshot[sr][name];idx=sm if name=='kv' else ss
    exp=source[:,:,sh].index_select(1,idx.to(source.device)).to(dev)
    actual=target[name][:,:,local].index_select(1,di if name=='kv' else dsi)
    if not torch.equal(actual,exp):raise AssertionError(f'{name} mismatch dr={dr}, head={glob}')
  for seg,total in enumerate([2048,2048,4096]):
   for sr in range(p):
    lo=max(sr*total//p,dr*total//d);hi=min((sr+1)*total//p,(dr+1)*total//d)
    if lo>=hi:continue
    prefix=sum([2048,2048,4096][:seg]);slo=prefix//p+lo-sr*total//p;dlo=prefix//d+lo-dr*total//d
    source=snapshot[sr]['conv']
    expected=source[:,:,:,slo:slo+hi-lo].index_select(1,ss.to(source.device)).to(dev)
    actual=target['conv'][:,:,:,dlo:dlo+hi-lo].index_select(1,dsi)
    if not torch.equal(actual,expected):raise AssertionError(f'conv mismatch segment={seg},dr={dr},sr={sr}')
  for name,t in target.items():
   used=dm if name=='kv' else ds
   mask=torch.ones(t.shape[1],dtype=torch.bool);mask[used]=False
   for ids in mask.nonzero().flatten().split(64):
    if not bool((t.index_select(1,ids.to(dev))==-777).all()):raise AssertionError(f'overwritten hole {name}')
 # Selected final-page tails are allowed to be copied; each page here includes payload
 # in its tail, enabling byte-exact full-page validation without forwarding the model.

def snapshot_source(plan):
 return [{name:t.clone() for name,t in r.items()} for r in plan['src']]

def check_source(plan,snapshot):
 for original,expected in zip(plan['src'],snapshot):
  for name,t in original.items():
   if not torch.equal(t,expected[name]):raise AssertionError('Source changed: '+name)

def candidate_view(plan):
 """Candidate receives private metadata copies and tensor aliases, never Judge state.
 Tensor detach() gives independent shape/stride metadata while sharing payload storage.
 """
 view={key:value for key,value in plan.items() if key not in ('src','dst','src_pages','dst_pages','src_slots','dst_slots')}
 for key in ('src_pages','dst_pages','src_slots','dst_slots'):view[key]=plan[key].clone()
 for group in ('src','dst'):
  view[group]=[{name:t.detach() for name,t in rank.items()} for rank in plan[group]]
 return view

def validate_candidate_view(view, trusted):
 """Reject map/scalar/tensor-descriptor mutation, using only evaluator-owned state."""
 if type(view) is not dict or view.keys()!=trusted.keys():raise AssertionError('Plan structure changed')
 for key in ('p','d','bs','ctx','cap','slotcap'):
  if type(view[key]) is not int or view[key]!=trusted[key]:raise AssertionError('Plan scalar changed: '+key)
 for key in ('src_pages','dst_pages','src_slots','dst_slots'):
  given,expected=view[key],trusted[key]
  if type(given) is not torch.Tensor or given.device!=expected.device or given.dtype!=expected.dtype or given.shape!=expected.shape or not torch.equal(given,expected):raise AssertionError('Plan map changed: '+key)
 for group in ('src','dst'):
  if type(view[group]) is not list or len(view[group])!=len(trusted[group]):raise AssertionError('Plan ranks changed')
  for given_rank,expected_rank in zip(view[group],trusted[group]):
   if type(given_rank) is not dict or given_rank.keys()!=expected_rank.keys():raise AssertionError('Plan state keys changed')
   for name,expected in expected_rank.items():
    given=given_rank[name]
    if type(given) is not torch.Tensor or given.dtype!=expected.dtype or given.device!=expected.device or given.shape!=expected.shape or given.stride()!=expected.stride() or given.storage_offset()!=expected.storage_offset() or given.data_ptr()!=expected.data_ptr():raise AssertionError('Plan tensor descriptor changed: '+name)

def load_candidate(path):
 spec=importlib.util.spec_from_file_location('candidate',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.transfer

def run_case(fn,p,d,bs,ctx,layout,warmup,samples,seed):
 plan=allocate(p,d,bs,ctx,layout=='fragmented',seed)
 try:
  for generation in [1,2]:
   fill(plan,generation);sync();before=snapshot_source(plan)
   exposed=candidate_view(plan)
   fn(exposed);sync();validate_candidate_view(exposed,plan);check(plan,before)
   check_source(plan,before);del before
   # Next generation permutes maps again, including slot reuse.
   plan.update(maps(bs,ctx,layout=='fragmented',seed+generation+1337))
  times=[]
  for i in range(warmup+samples):
   plan.update(maps(bs,ctx,layout=='fragmented',seed+i+8192));fill(plan,i+3);before=snapshot_source(plan);sync()
   exposed=candidate_view(plan)
   start=time.perf_counter_ns();fn(exposed);sync();elapsed=(time.perf_counter_ns()-start)/1000
   if i>=warmup:times.append(elapsed)
   # Check the measured result too; every measured invocation has new data/maps.
   validate_candidate_view(exposed,plan);check(plan,before);check_source(plan,before);del before,exposed
  allocated=[torch.cuda.max_memory_allocated(i) for i in range(4)]
  physical=2*bs*math.ceil(ctx/BLOCK)*BLOCK*2*256*2*10+bs*64471040
  return dict(p=p,d=d,bs=bs,ctx=ctx,layout=layout,correct=True,median_us=statistics.median(times),samples_us=times,useful_bytes=bs*(ctx*20480+64471040),page_payload_bytes=physical,peak_allocated=allocated)
 finally:
  del plan;sync();torch.cuda.empty_cache()

def main():
 a=argparse.ArgumentParser();a.add_argument('--candidate');a.add_argument('--output',required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--warmup',type=int,default=3);a.add_argument('--samples',type=int,default=10);args=a.parse_args()
 if torch.cuda.device_count()!=4:raise RuntimeError('Require exactly four delegated GPUs')
 for i in range(4):
  for j in range(4):
   if i!=j and not torch.cuda.can_device_access_peer(i,j):raise RuntimeError(f'No peer access {i}->{j}')
 fn=reference if not args.candidate else load_candidate(args.candidate)
 output=Path(args.output);output.parent.mkdir(exist_ok=True,parents=True)
 env={'torch':torch.__version__,'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_properties(i).__dict__ if hasattr(torch.cuda.get_device_properties(i),'__dict__') else str(torch.cuda.get_device_properties(i)) for i in range(4)],'mode':'torch-reference' if not args.candidate else args.candidate,'scope':'synthetic transfer kernels only; no model forward; no vLLM IPC or NIXL service overhead','block_size':BLOCK,'started_at':time.time()}
 (output.with_suffix('.environment.json')).write_text(json.dumps(env,indent=2,default=str))
 with output.open('w') as out:
  for p,d in [(1,1),(1,2),(2,1),(2,2)]:
   for ctx in ([4097] if args.smoke else LENGTHS):
    for bs in ([1] if args.smoke else [1,4]):
     for layout in ['contiguous','fragmented']:
      for i in range(4):torch.cuda.reset_peak_memory_stats(i)
      row=run_case(fn,p,d,bs,ctx,layout,args.warmup,args.samples,seed=12345+ctx+bs)
      out.write(json.dumps(row)+'\n');out.flush();print(json.dumps(row),flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
