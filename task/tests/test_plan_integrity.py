"""Regression cases for the concrete mutable-plan correctness bypass."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment'))
sys.path.insert(0,'/opt/kv-transfer')
import benchmark
class PlanIntegrity(unittest.TestCase):
 def plan(self):
  p={'p':1,'d':1,'bs':1,'ctx':129,'cap':6,'slotcap':4}
  for k in ('src_pages','dst_pages'):p[k]=torch.tensor([2,0],dtype=torch.int64)
  for k in ('src_slots','dst_slots'):p[k]=torch.tensor([1],dtype=torch.int64)
  for g in ('src','dst'):p[g]=[{k:torch.zeros((3,4)) for k in ('kv','conv','recurrent')}]
  return p
 def test_valid_alias_view(self):
  p=self.plan();benchmark.validate_candidate_view(benchmark.candidate_view(p),p)
 def test_empty_rebound_maps(self):
  p=self.plan();v=benchmark.candidate_view(p)
  for k in ('src_pages','dst_pages','src_slots','dst_slots'):v[k]=torch.empty(0,dtype=torch.int64)
  with self.assertRaises(AssertionError):benchmark.validate_candidate_view(v,p)
  self.assertEqual(p['src_pages'].numel(),2)
 def test_inplace_map_resize(self):
  p=self.plan();v=benchmark.candidate_view(p);v['src_pages'].resize_(0)
  with self.assertRaises(AssertionError):benchmark.validate_candidate_view(v,p)
  self.assertEqual(p['src_pages'].tolist(),[2,0])
 def test_inplace_map_values(self):
  p=self.plan();v=benchmark.candidate_view(p);v['src_pages'].zero_()
  with self.assertRaises(AssertionError):benchmark.validate_candidate_view(v,p)
 def test_scalar_mutation(self):
  p=self.plan();v=benchmark.candidate_view(p);v['bs']=0
  with self.assertRaises(AssertionError):benchmark.validate_candidate_view(v,p)
 def test_tensor_shape_mutation(self):
  p=self.plan();v=benchmark.candidate_view(p);v['dst'][0]['kv'].resize_(0)
  with self.assertRaises(AssertionError):benchmark.validate_candidate_view(v,p)
  self.assertEqual(tuple(p['dst'][0]['kv'].shape),(3,4))
 def test_tensor_rebinding(self):
  p=self.plan();v=benchmark.candidate_view(p);v['dst'][0]['kv']=torch.empty_like(v['dst'][0]['kv'])
  with self.assertRaises(AssertionError):benchmark.validate_candidate_view(v,p)
if __name__=='__main__':unittest.main()
