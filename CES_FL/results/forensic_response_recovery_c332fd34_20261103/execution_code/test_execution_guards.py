import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class Guards(unittest.TestCase):
    def test_remote_launch_code_is_valid_and_quotes_literal_arguments(self):
        m=load('study_coordinator');captured=[]
        def run(cmd,**kw):
            captured.append(cmd)
            remote=shlex.split(cmd[-1]);self.assertEqual(remote[:2],['python3','-c'])
            ast.parse(remote[2])
            return subprocess.CompletedProcess(cmd,0,'12345\n','')
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)),patch.object(m,'state'),patch.object(m.subprocess,'run',side_effect=run):
            m.spawn('gpu01','fixture',['--dataset','ffpp','--mode','arm','--arm','A'])
        self.assertEqual(len(captured),1)

    def test_existing_launch_record_does_not_spawn_duplicate(self):
        m=load('study_coordinator')
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)),patch.object(m.subprocess,'run') as run:
            (Path(td)/'fixture_remote.pid').write_text('12345\n')
            m.spawn('gpu01','fixture',[])
            run.assert_not_called()

    def test_project_budget_rejects_overcommit_without_gpu_or_file_changes(self):
        m=load('direct_stage')
        d={'initial_total_gpu_hours':24.,'gpu_sessions':[dict(id='existing',state='RUNNING',reserved_gpu_hours=23.5,elapsed_seconds=1.)]}
        @contextmanager
        def fake_ledger():yield d
        with patch.object(m.budget,'ledger',fake_ledger):
            with self.assertRaisesRegex(RuntimeError,'budget'):m.reserve('new',.6)
        self.assertEqual(len(d['gpu_sessions']),1)

    def test_full_dataset_qualification_is_required_for_counterfactual_role(self):
        m=load('direct_stage')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):m.readiness(Path(td))


if __name__=='__main__':unittest.main(verbosity=2)
