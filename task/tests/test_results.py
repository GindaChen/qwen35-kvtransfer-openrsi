"""Malformed result files cannot become scores (not an authenticity boundary)."""
import itertools
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import evaluate
from evaluate import load

def rows():
    return [dict(p=p,d=d,bs=bs,ctx=ctx,layout=layout,correct=True,
                 median_us=2.0,samples_us=[2.0]*50)
        for (p,d),bs,ctx,layout in itertools.product(
            [(1,1),(1,2),(2,1),(2,2)],[1,4],
            [4096,8192,32768,65536,4097,10034,32749,65521],
            ['contiguous','fragmented','fragmented_runs'])]

class Results(unittest.TestCase):
    def read(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'results.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in data))
            return load(p)
    def test_complete_suite(self):
        self.assertEqual(len(self.read(rows())),192)
    def test_scoring_refuses_without_isolator(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'verifier'
            with mock.patch.object(evaluate,'LOG',log), mock.patch.object(evaluate,'ISOLATOR',str(Path(tmp)/'absent')):
                with self.assertRaisesRegex(RuntimeError,'rsi-isolate is unavailable'):
                    evaluate.main()
            self.assertFalse((log/'reward.json').exists())
            self.assertEqual(json.loads((log/'failure.json').read_text())['status'],'unscored')
    def test_scoring_refuses_unconfined_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib=Path(tmp)/'iso';lib.mkdir()
            (lib/'rsi_isolate.py').write_text('import json;print(json.dumps({"mode":"same_user","fallback":"same-user"}))')
            log=Path(tmp)/'verifier'
            with mock.patch.object(evaluate,'LOG',log), mock.patch.object(evaluate,'ISOLATOR',str(lib)):
                with self.assertRaisesRegex(RuntimeError,'Unconfined'):
                    evaluate.main()
            self.assertFalse((log/'reward.json').exists())
    def test_invalid(self):
        for change in ['duplicate','missing','truthy','median','nan','negative','count']:
            with self.subTest(change=change):
                data=rows()
                if change=='duplicate': data[-1]=data[0].copy()
                if change=='missing': data.pop()
                if change=='truthy': data[0]['correct']='false'
                if change=='median': data[0]['median_us']=0.001
                if change=='nan': data[0]['samples_us'][0]=float('nan')
                if change=='negative': data[0]['samples_us'][0]=-1
                if change=='count': data[0]['samples_us'].pop()
                with self.assertRaises(RuntimeError): self.read(data)

if __name__=='__main__': unittest.main()
