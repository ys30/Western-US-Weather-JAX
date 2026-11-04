"""Regression checks for vectorized calendar-day paired RMSE bootstrap.

Verifies exact quantiles against a naive row-level resampling reference,
including a sparse observed-hot subset and missing calendar dates.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_paired_station_predictions import daily_block_interval

def brute(frame,days,draws,seed,threshold=None):
    days_all=pd.date_range(frame.timestamp_utc.min().floor("D"),
                           frame.timestamp_utc.max().floor("D"),freq="D")
    groups={date:g for date,g in frame.groupby(frame.timestamp_utc.dt.floor("D"))}
    rng=np.random.default_rng(seed)
    n=len(days_all)
    values=[]
    missing=0
    for _ in range(draws):
        starts=rng.integers(0,n,size=(n+days-1)//days)
        indices=[(int(start)+offset)%n for start in starts for offset in range(days)][:n]
        selected=[groups[days_all[idx]] for idx in indices if days_all[idx] in groups]
        if not selected:
            missing+=1;continue
        picked=pd.concat(selected,ignore_index=True)
        if threshold is not None:
            picked=picked.loc[picked.observed_K>=threshold]
        if picked.empty:
            missing+=1;continue
        h=np.sqrt(np.mean((picked.historical_K-picked.observed_K)**2))
        s=np.sqrt(np.mean((picked.source_mean_K-picked.observed_K)**2))
        values.append(s-h)
    return (np.quantile(values,[.025,.975]) if len(values)>=max(10,int(.8*draws))
            else None),missing

def test():
    dates=pd.date_range("2022-07-01",periods=22,freq="D",tz="UTC")
    chosen=np.array([i for i in range(len(dates)) if i not in (4,8,13)])
    times=[dates[i]+pd.Timedelta(hours=h) for i in chosen for h in (0,6,12,18)]
    n=len(times)
    rng=np.random.default_rng(123)
    frame=pd.DataFrame({"timestamp_utc":pd.to_datetime(times,utc=True),
        "observed_K":300+rng.normal(0,3,n),
        "historical_K":300+rng.normal(0,3,n),
        "source_mean_K":300+rng.normal(0,3,n)})
    frame.loc[frame.timestamp_utc.dt.day.isin((2,3,7,9,10,18,20,21)),"observed_K"]=306.
    for threshold in (None,305.):
        expected,empty=brute(frame,days=3,draws=100,seed=45,threshold=threshold)
        measured=daily_block_interval(frame,days=3,draws=100,seed=45,hot_threshold=threshold)
        assert measured is not None and expected is not None
        assert measured["empty_draws"]==empty
        assert np.allclose([measured["ci95_low_K"],measured["ci95_high_K"]],
                           expected,rtol=0,atol=1e-12)
    print("Fast block-bootstrap matches naive row-level RMSE quantiles")

if __name__=="__main__":
    test()
