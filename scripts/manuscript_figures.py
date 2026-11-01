"""Build paper figure assets ONLY from successful CI JSON artifacts."""
import argparse,csv,json,html
from pathlib import Path

def records(folder,keys):
    files=list(Path(folder).rglob("seed-*.json"))
    data={}
    for path in files:
        stem=path.stem
        seed=int(stem.rsplit("-",1)[-1])
        if seed in data:raise ValueError("Duplicate seed")
        v=json.loads(path.read_text())
        # The frozen-region artifact contains BOTH adapted and source-global-constant
        # ablation results. Keep only the historical-climatology-adapted protocol.
        if "fixed_305k_cnn_rmse_kelvin" in keys and v.get("climatology_mode") != "target_history":
            continue
        if seed in data:raise ValueError(f"Duplicate seed in selected protocol: {seed}")
        data[seed]=[float(v[k]) for k in keys]
    if set(data)!={7,42,123}:raise ValueError(f"Unexpected seeds: {set(data)}")
    return data

def chart(name,rows,path):
    colors=("#245d9a","#de772e")
    maxv=max(v for pair in rows.values() for v in pair)*1.12
    x=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 350">',
       '<rect width="800" height="350" fill="#ffffff"/>',
       f'<title>{html.escape(name)}</title>',
       f'<text x="26" y="36" font-family="sans-serif" font-size="20">{html.escape(name)}</text>']
    for i,(seed,pair) in enumerate(sorted(rows.items())):
        y=70+i*80
        x.append(f'<text x="26" y="{y+22}" font-size="15" font-family="sans-serif">Seed {seed}</text>')
        for j,v in enumerate(pair):
            width=460*v/maxv
            x.append(f'<rect x="140" y="{y+25*j}" width="{width:.1f}" height="20" fill="{colors[j]}"/>')
            x.append(f'<text x="{150+width:.1f}" y="{y+25*j+15}" font-family="sans-serif" font-size="13">{v:.3f} K</text>')
    x.append('<text x="26" y="330" font-family="sans-serif" font-size="12">Blue: CNN; orange: protocol-specific reference. Lower RMSE is better.</text>')
    x.append('</svg>')
    path.write_text("\n".join(x))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--adapted",required=True);p.add_argument("--source-only",required=True)
    p.add_argument("--output",required=True);args=p.parse_args()
    ak=("test_rmse_kelvin","source_weight_blend_rmse_kelvin","fixed_305k_cnn_rmse_kelvin","fixed_305k_blend_rmse_kelvin")
    sk=("target_annual_cnn_rmse_kelvin","target_annual_persistence_rmse_kelvin","target_hot_cnn_rmse_kelvin","target_hot_persistence_rmse_kelvin")
    a=records(args.adapted,ak);s=records(args.source_only,sk)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    chart("Frozen CNN with adapted climatology: hot RMSE",{k:v[2:] for k,v in a.items()},out/"adapted_hot.svg")
    chart("Climatology-free source CNN: hot RMSE",{k:v[2:] for k,v in s.items()},out/"source_only_hot.svg")
    with (out/"comparison.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("protocol","seed","cnn_annual_K","reference_annual_K","cnn_observed_hot_K","reference_observed_hot_K"))
        for label,rows in (("adapted",a),("source_only",s)):
            for seed,values in sorted(rows.items()):w.writerow((label,seed,*values))
    (out/"README.md").write_text("# Artifact-derived paper figures\n\nInput runs:\n"
        "- Frozen climatology-adapted: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41279255173\n"
        "- Source-only: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41277714617\n\n"
        "Do not conflate these protocols' different benchmarks. Truth-hot RMSE is retrospective.\n")
if __name__=="__main__":main()
