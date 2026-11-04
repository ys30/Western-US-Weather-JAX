"""Render an evidence-gated SVG from verified ASOS station metric CSV only.

This is a descriptive six-point display (2 stations x 3 seeds).
Intervals reflect temporal resampling conditional on each fitted pair,
NOT confidence intervals across seeds or independent operational forecasts.
"""
import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

SEEDS=(42,7,123)
SITES=("ABQ","ROW")

def load(path):
    with Path(path).open(newline="") as fp:
        rows=list(csv.DictReader(fp))
    keys={(r["station"],int(r["seed"])) for r in rows}
    expected={(site,seed) for site in SITES for seed in SEEDS}
    if len(rows)!=6 or keys!=expected:
        raise ValueError("Expected exactly 2 stations x 3 seeds")
    parsed=[]
    for row in rows:
        n=int(row["matched_n"])
        cover=float(row["match_rate"])
        if n<1000 or cover<0.8:
            raise ValueError("Insufficient ASOS coverage")
        vals={k:float(row[k]) for k in
              ("historical_rmse_K","source_mean_rmse_K","baseline_rmse_K",
               "delta_source_mean_minus_historical_rmse_K",
               "annual_ci95_low_K","annual_ci95_high_K")}
        if any(not math.isfinite(v) for v in vals.values()):
            raise ValueError("Missing or nonfinite model/CI value")
        delta=vals["source_mean_rmse_K"]-vals["historical_rmse_K"]
        if abs(delta-vals["delta_source_mean_minus_historical_rmse_K"])>1e-6:
            raise ValueError("Paired RMSE arithmetic mismatch")
        if not vals["annual_ci95_low_K"]<=vals["annual_ci95_high_K"]:
            raise ValueError("Inverted interval")
        parsed.append({"site":row["station"],"seed":int(row["seed"]),
                       "n":n,**vals})
    return sorted(parsed,key=lambda d:(SITES.index(d["site"]),SEEDS.index(d["seed"])))

def draw(rows):
    left,right=235,910
    # Symmetric axis intentionally includes zero even if all contrasts share sign.
    edge=max(0.025,*[abs(r[k]) for r in rows for k in
                     ("annual_ci95_low_K","annual_ci95_high_K",
                      "delta_source_mean_minus_historical_rmse_K")])
    edge*=1.12
    x=lambda v: left+(v+edge)/(2*edge)*(right-left)
    y=lambda i:120+i*53+(26 if i>=3 else 0)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="540" viewBox="0 0 980 540" role="img" aria-label="Observed ASOS paired RMSE differences with 14 day block bootstrap intervals">']
    out.append('<rect width="980" height="540" fill="white"/>')
    out.append('<text x="34" y="42" font-size="23" font-family="sans-serif" font-weight="bold" fill="#17212b">Southwest 2022: ASOS station comparison</text>')
    out.append('<text x="34" y="68" font-size="14" font-family="sans-serif" fill="#475569">Source-mean minus historical RMSE (K) · positive favors historical</text>')
    out.append(f'<line x1="{x(0):.2f}" y1="92" x2="{x(0):.2f}" y2="406" stroke="#64748b" stroke-dasharray="5 5"/>')
    out.append('<line x1="32" y1="273" x2="940" y2="273" stroke="#e2e8f0"/>')
    for i,r in enumerate(rows):
        yi=y(i); lo=x(r["annual_ci95_low_K"]);hi=x(r["annual_ci95_high_K"])
        mid=x(r["delta_source_mean_minus_historical_rmse_K"])
        site=escape(r["site"])
        out.append(f'<text x="34" y="{yi+5}" font-size="15" font-family="sans-serif" fill="#17212b">{site} · seed {r["seed"]}</text>')
        out.append(f'<line x1="{lo:.2f}" y1="{yi}" x2="{hi:.2f}" y2="{yi}" stroke="#287663" stroke-width="3"/>')
        for at in (lo,hi):
            out.append(f'<line x1="{at:.2f}" y1="{yi-6}" x2="{at:.2f}" y2="{yi+6}" stroke="#287663" stroke-width="2"/>')
        out.append(f'<circle cx="{mid:.2f}" cy="{yi}" r="6" fill="#145b88" stroke="white" stroke-width="1.5"/>')
        out.append(f'<text x="916" y="{yi+5}" font-size="12" font-family="sans-serif" fill="#475569" text-anchor="end">n={r["n"]}</text>')
    for v in (-edge,0,edge):
        out.append(f'<text x="{x(v):.2f}" y="448" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#334155">{v:+.3f}</text>')
    out.append('<text x="35" y="484" font-size="13" font-family="sans-serif" fill="#475569">Dots: paired RMSE difference; whiskers: within-seed 14-day calendar-block bootstrap 95% interval.</text>')
    out.append('<text x="35" y="505" font-size="13" font-family="sans-serif" fill="#475569">Retrospective station agreement only; IEM reports may also inform ERA5 assimilation.</text>')
    out.append('<text x="35" y="526" font-size="12" font-family="sans-serif" fill="#64748b">Not an operational forecast score; three random seeds are not three independent climate samples.</text>')
    out.append('</svg>')
    return "\n".join(out)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metrics-csv",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    rows=load(args.metrics_csv)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(draw(rows))
    print("Evidence-gated ASOS descriptive figure:",path)

if __name__=="__main__":
    main()
