"""Retrospective truth-defined multiday regional heat episode diagnostics."""
import numpy as np

def event_diagnostics(pred, baseline, truth, target_times, threshold_kelvin=305.0,
                      min_region_fraction=0.05, min_days=2):
    p=np.asarray(pred); b=np.asarray(baseline); y=np.asarray(truth)
    ts=np.asarray(target_times).astype("datetime64[D]")
    if p.shape!=b.shape or p.shape!=y.shape or y.ndim!=3 or len(ts)!=len(y):
        raise ValueError("Expected aligned [time,lat,lon] arrays and times")
    if not 0<min_region_fraction<=1 or min_days<1:raise ValueError("Invalid event criteria")
    days=np.unique(ts)
    # Event indicator on each observed day: any valid time with >=5% grid area hot.
    active=np.array([np.any(np.mean(y[ts==day]>=threshold_kelvin,axis=(1,2))>=min_region_fraction) for day in days])
    events=[];start=None
    def close(end):
        selected=days[start:end]
        if len(selected)<min_days:return
        idx=np.isin(ts,selected)
        hot=y[idx]>=threshold_kelvin
        if not hot.any():return
        score=lambda arr,mask:float(np.sqrt(np.mean((arr[mask]-y[idx][mask])**2)))
        events.append({"start_date":str(selected[0]),"end_date":str(selected[-1]),
                       "days":int(len(selected)),"valid_times":int(idx.sum()),
                       "hot_gridcell_times":int(hot.sum()),
                       "cnn_hot_rmse_kelvin":score(p[idx],hot),
                       "blend_hot_rmse_kelvin":score(b[idx],hot),
                       "cnn_all_rmse_kelvin":score(p[idx],np.ones_like(hot,dtype=bool)),
                       "blend_all_rmse_kelvin":score(b[idx],np.ones_like(hot,dtype=bool))})
    for i in range(len(days)+1):
        flag=(i<len(days) and active[i])
        if flag and start is None:start=i
        if start is not None and (not flag or (i>start and days[i]-days[i-1]>np.timedelta64(1,"D"))):
            close(i);start=i if flag else None
    return {"definition":"Truth-defined retrospective regional episode: >=5% cells reach 305 K at any 6-hour valid time on >=2 consecutive days",
            "not_operational_event_detection":True,"events":events,
            "episode_count":len(events),
            "episodes_cnn_better_hot":sum(e["cnn_hot_rmse_kelvin"]<e["blend_hot_rmse_kelvin"] for e in events)}


def event_definition_sensitivity(pred, baseline, truth, target_times,
                                 fractions=(0.01, 0.05, 0.10),
                                 durations=(2, 3, 4),
                                 threshold_kelvin=305.0):
    """Prespecified 3×3 retrospective event-definition grid; no selection.

    Each result reports its own qualifying event count. Undefined fractions
    (no qualifying episodes) are represented by None rather than zero.
    """
    results=[]
    for fraction in fractions:
        for days in durations:
            report=event_diagnostics(pred, baseline, truth, target_times,
                                     threshold_kelvin=threshold_kelvin,
                                     min_region_fraction=fraction,min_days=days)
            n=report["episode_count"]
            wins=report["episodes_cnn_better_hot"]
            results.append({"min_region_fraction":float(fraction),
                            "min_consecutive_days":int(days),
                            "episode_count":n,
                            "episodes_cnn_better_hot":wins,
                            "episode_win_fraction":wins/n if n else None})
    return {"definition":"Retrospective observed-temperature events; fixed 305 K threshold",
            "not_operational_event_detection":True,
            "prespecified_sensitivity_grid":results}
