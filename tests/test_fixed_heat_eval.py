import json
import subprocess
import sys

def test_fixed_hot_evaluation_cli(tmp_path):
    output=tmp_path/"group"
    p=subprocess.run([sys.executable,"-m","weatherjax","--epochs","1","--channels","4","--group-metrics","--hot-threshold-kelvin","305","--output",str(output)],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+"\n"+p.stderr
    metrics=json.loads((output/"metrics.json").read_text())
    heat=metrics["fixed_heat_evaluation"]
    assert heat["threshold_kelvin"]==305
    assert heat["gridcell_count"]>=0
