"""Explicit, reproducible timestamp and input-availability audit for ERA5 24h leads."""
import numpy as np

def audit_timeline(times, train_end, validation_end, lead_steps=4, history_steps=4, step_hours=6):
    ts=np.asarray(times).astype("datetime64[h]")
    if ts.ndim!=1 or len(ts)<20 or np.isnat(ts).any():
        raise ValueError("Invalid timestamps")
    diffs=np.diff(ts).astype("timedelta64[h]").astype(int)
    if np.any(diffs!=step_hours):
        raise ValueError("Missing, duplicate or irregular forecast timestamps")
    a=int(np.searchsorted(ts,np.datetime64(train_end)))
    b=int(np.searchsorted(ts,np.datetime64(validation_end)))
    if not (lead_steps+history_steps-1<a<b<len(ts)):
        raise ValueError("Invalid chronological training/validation/test boundaries")
    offsets=np.arange(history_steps-1,-1,-1)
    summary={}
    for name,start,end in (("train",lead_steps+history_steps-1,a),
                           ("validation",a,b),("test",b,len(ts))):
        targets=np.arange(start,end)
        inputs=targets[:,None]-lead_steps-offsets
        issue=targets-lead_steps
        if np.any(inputs>issue[:,None]) or np.any(inputs<0):
            raise AssertionError("Lookahead or invalid input indices")
        # Validation/test target timestamps cannot participate in any training target.
        if name=="train" and np.any(targets>=a):
            raise AssertionError("Training target beyond boundary")
        summary[name]={"samples":int(len(targets)),
            "first_target":str(ts[targets[0]]),
            "last_target":str(ts[targets[-1]]),
            "first_input":str(ts[inputs[0,0]]),
            "last_input":str(ts[inputs[-1,-1]]),
            "target_step_minus_latest_input":int(lead_steps)}
    return {"lead_hours":lead_steps*step_hours,"cadence_hours":step_hours,
            "train_end_exclusive":str(ts[a]),"validation_end_exclusive":str(ts[b]),
            "splits":summary,"max_lookahead_hours":0}

def assert_source_only_inputs(target_indices, length, lead_steps=4, history_steps=4):
    t=np.asarray(target_indices,dtype=int)
    idx=t[:,None]-lead_steps-np.arange(history_steps-1,-1,-1)[None,:]
    if np.any(idx<0) or np.any(idx+lead_steps>t[:,None]) or np.any(t>=length):
        raise ValueError("Future input or bad forecast target indices")
    return idx
