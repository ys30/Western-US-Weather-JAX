"""No-lookahead 24-hour temperature prediction datasets."""
import numpy as np

def synthetic_data(n=240, h=20, w=24, seed=42):
    """Smooth synthetic spatiotemporal field; for software smoke tests ONLY."""
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:h,:w]
    spatial=(2*np.sin(x/5)+3*np.cos(y/7)).astype(np.float32)
    t=np.arange(n,dtype=np.float32)
    seasonal=5*np.sin(2*np.pi*t/80)
    trend=.015*t
    field=(285+spatial[None]+seasonal[:,None,None]+trend[:,None,None]
           +rng.normal(0,.3,(n,h,w))).astype('float32')
    return field

def load_era5(path, variable='t2m', lat_min=30, lat_max=50, lon_min=-125, lon_max=-102):
    """Read pre-downloaded ERA5 NetCDF / Zarr with hourly time, lat, lon."""
    import xarray as xr
    ds=xr.open_zarr(path) if str(path).endswith('.zarr') else xr.open_dataset(path)
    if variable not in ds: raise ValueError(f'Missing {variable}; variables={list(ds.data_vars)}')
    da=ds[variable]
    if 'latitude' in da.dims: da=da.rename({'latitude':'lat'})
    if 'longitude' in da.dims: da=da.rename({'longitude':'lon'})
    if 'valid_time' in da.dims: da=da.rename({'valid_time':'time'})
    if not set(('time','lat','lon')).issubset(da.dims): raise ValueError('Expected time/lat/lon dimensions')
    if da.lon.max()>180: da=da.assign_coords(lon=((da.lon+180)%360)-180).sortby('lon')
    da=da.sortby('lat').sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max)).transpose('time','lat','lon')
    if da.sizes['lat']==0 or da.sizes['lon']==0: raise ValueError('Empty region after crop')
    return da

def prepare_splits(arr, lead_steps=4, train_fraction=.7, val_fraction=.15):
    """Chronological disjoint TARGET periods. Input at t; target at t+lead_steps."""
    arr=np.asarray(arr,dtype=np.float32)
    if arr.ndim!=3 or not np.isfinite(arr).all(): raise ValueError('Expected finite [time,height,width] array')
    n=len(arr); a=int(n*train_fraction); b=int(n*(train_fraction+val_fraction))
    if not (0<lead_steps<a<b<n): raise ValueError('Insufficient steps')
    mean=float(arr[:a].mean()); std=float(arr[:a].std()); std=max(std,1e-6)
    def make(lo,hi):
        targets=np.arange(max(lo,lead_steps),hi)
        x=(arr[targets-lead_steps]-mean)/std
        y=(arr[targets]-mean)/std
        return x[...,None],y[...,None]
    return (make(lead_steps,a),make(a,b),make(b,n)),(mean,std)
