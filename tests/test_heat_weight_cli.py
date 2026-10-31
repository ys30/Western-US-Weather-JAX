import subprocess
import sys
import json

def test_cli_heat_weight_works(tmp_path):
    out=tmp_path/"heat"
    cp=subprocess.run([sys.executable,"-m","weatherjax","--epochs","1","--channels","4",
        "--hot-weight","2","--hot-threshold-kelvin","305","--output",str(out)],
        capture_output=True,text=True,timeout=120)
    assert cp.returncode==0,cp.stdout+"\n"+cp.stderr
    metrics=json.loads((out/"metrics.json").read_text())
    assert metrics["hot_weight"]==2
    assert metrics["hot_threshold_kelvin"]==305
