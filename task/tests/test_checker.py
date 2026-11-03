"""Exercise selected data, holes, source snapshots and stale outputs on CPU."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment'))
import benchmark as b

class Checker(unittest.TestCase):
    def setUp(self):
        # The checker is independent of page payload dimensions. Keep the real
        # head/channel counts while using small page/recurrent payloads on CPU.
        self.plan = dict(p=1,d=1,bs=1,ctx=129,cap=4,slotcap=4,
            src_pages=torch.tensor([2,0]),dst_pages=torch.tensor([1,3]),
            src_slots=torch.tensor([2]),dst_slots=torch.tensor([1]))
        self.plan['src']=[dict(kv=torch.randn(10,4,2,2,2).bfloat16(),
            conv=torch.randn(30,4,3,8192).bfloat16(),
            recurrent=torch.randn(30,4,32,2,2))]
        self.plan['dst']=[{k:torch.full_like(v,333) for k,v in self.plan['src'][0].items()}]
        self.snapshot=b.snapshot_source(self.plan)
        self.copy()
    def copy(self):
        for name,src in self.plan['src'][0].items():
            sm=self.plan['src_pages'] if name=='kv' else self.plan['src_slots']
            dm=self.plan['dst_pages'] if name=='kv' else self.plan['dst_slots']
            self.plan['dst'][0][name].index_copy_(1,dm,src.index_select(1,sm))
    def test_complete_transfer(self):
        b.check(self.plan,self.snapshot,333)
        b.check_source(self.plan,self.snapshot)
    def test_each_omitted_family(self):
        for name in self.plan['dst'][0]:
            with self.subTest(name=name):
                self.copy()
                self.plan['dst'][0][name].fill_(333)
                with self.assertRaises(AssertionError): b.check(self.plan,self.snapshot,333)
    def test_stale_output(self):
        for t in self.plan['src'][0].values(): t.add_(1)
        self.snapshot=b.snapshot_source(self.plan)
        with self.assertRaises(AssertionError): b.check(self.plan,self.snapshot,333)
    def test_source_mutation(self):
        self.plan['src'][0]['conv'].zero_()
        with self.assertRaises(AssertionError): b.check_source(self.plan,self.snapshot)
    def test_hole_write(self):
        self.plan['dst'][0]['recurrent'][:,0].zero_()
        with self.assertRaises(AssertionError): b.check(self.plan,self.snapshot,333)

if __name__=='__main__': unittest.main()
