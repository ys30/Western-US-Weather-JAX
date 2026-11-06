"""Plot the study geography from the archived ERA5 data used for the paper.

Figure 1(b) style, updated with legible map labels. This does NOT simulate
temperature: the raster must come from the archived source/target NetCDF.
The output is a public, paper-consistent geographic visualization.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr

DOMAINS = {
    "source": ("source.nc", (-121, -109, 31, 43), "#1c72a9"),
    "southwest": ("southwest.nc", (-108, -102, 30, 36), "#009e73"),
    "plains": ("plains.nc", (-101, -95, 32, 38), "#d55e00"),
}


def summer_mean(path, expected_box):
    with xr.open_dataset(path) as dataset:
        if "t2m" not in dataset:
            raise ValueError(f"{path}: t2m absent")
        t = dataset.t2m
        dates = t.time.dt
        summer = t.where(
            dates.year.isin([2019, 2020]) & dates.month.isin([6, 7, 8]),
            drop=True,
        ).mean("time").load()
        if summer.isnull().any().item():
            raise ValueError(f"Missing summer temperatures in {path}")
        lat, lon = summer.lat.values, summer.lon.values
        if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError(f"Nonascending coordinates: {path}")
        lo, hi, south, north = expected_box
        if (len(lat), len(lon)) not in [(49, 49), (25, 25)]:
            raise ValueError(f"Unexpected ERA5 grid shape: {path} {summer.shape}")
        if not (np.isclose(lon[0], lo) and np.isclose(lon[-1], hi) and
                np.isclose(lat[0], south) and np.isclose(lat[-1], north)):
            raise ValueError(f"Unexpected ERA5 grid bounds: {path}")
        return lon, lat, summer.values - 273.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.8, 7.3), dpi=160)
    fig.subplots_adjust(left=.075, right=.90, top=.93, bottom=.205)
    world = Basemap(projection="cyl", llcrnrlon=-126.5, llcrnrlat=27,
                    urcrnrlon=-93, urcrnrlat=48.5, resolution="l", ax=ax)
    world.drawmapboundary(fill_color="#f8fbfd", linewidth=.8, color="#bdc9d1")
    world.fillcontinents(color="#f9faf8", lake_color="#f8fbfd", zorder=0)
    world.drawcoastlines(linewidth=.68, color="#8d9ca6", zorder=1)
    world.drawcountries(linewidth=.65, color="#b4c0c7", zorder=1)
    world.drawstates(linewidth=.55, color="#b5bec7", zorder=1)

    image = None
    for name, (filename, box, color) in DOMAINS.items():
        lo, hi, south, north = box
        lon, lat, values = summer_mean(args.input / filename, box)
        # Exact 0.25-degree grid-centre coordinates; no spatial interpolation.
        image = ax.pcolormesh(lon, lat, values, cmap="RdYlBu_r",
                              shading="nearest", vmin=15, vmax=34,
                              rasterized=True, zorder=2)
        ax.add_patch(Rectangle((lo, south), hi-lo, north-south,
                               fill=False, edgecolor=color,
                               linewidth=2.7, zorder=4))
    # Labels positioned outside the data rectangles to preserve the temperature field.
    labels = [
        (-121.8, 44.0, "Source training\n2019–2020", "#1c72a9", "left"),
        (-108.0, 29.1, "Southwest target", "#009e73", "left"),
        (-101.4, 39.1, "Southern Plains target", "#bd5500", "left"),
    ]
    for lon, lat, label, color, horizontal in labels:
        ax.text(lon, lat, label, ha=horizontal, va="center", fontsize=9.3,
                color=color, weight="bold", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=1.7))
    # Grounded city coordinates are never shifted to improve appearance. Only
    # text-label display offsets change. Offsets are in screen points, not degrees.
    # The Albuquerque station is ~22.5% from the western edge of the SW box and
    # ~15.3% below its northern edge: (-106.6504, 35.0844).
    cities = [
        # city, longitude, latitude, text offset in points (x, y)
        ("Salt Lake City", -111.8910, 40.7608, (-52, 9)),
        ("Las Vegas", -115.1398, 36.1699, (7, 7)),
        ("Los Angeles", -118.2437, 34.0522, (7, -12)),
        ("Phoenix", -112.0740, 33.4484, (7, -14)),
        ("Albuquerque", -106.6504, 35.0844, (10, -17)),
        ("Denver", -104.9903, 39.7392, (8, 8)),
        ("Oklahoma City", -97.5164, 35.4676, (-15, -17)),
        ("Dallas", -96.7970, 32.7767, (-28, -14)),
    ]
    for name, x, y, offset in cities:
        # White ring distinguishes the geographic anchor from the ERA5 raster.
        ax.scatter([x], [y], s=29, facecolor="white", edgecolor="white",
                   linewidth=1.2, zorder=7)
        ax.scatter([x], [y], s=11, facecolor="#203746", edgecolor="none",
                   zorder=8)
        ax.annotate(name, xy=(x, y), xycoords="data",
                    xytext=offset, textcoords="offset points", fontsize=8.0,
                    color="#203746", fontweight="bold" if name=="Albuquerque" else "normal",
                    ha="right" if offset[0]<0 else "left",
                    va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-", lw=0.65, color="#617785",
                                    shrinkA=2.0, shrinkB=2.0),
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=.88, pad=1.0))
    ax.set_xlim(-126.5, -93)
    ax.set_ylim(27, 48.5)
    ax.set_xticks([-125, -120, -115, -110, -105, -100, -95])
    ax.set_yticks([30, 35, 40, 45])
    ax.set_xticklabels(["125°W", "120°W", "115°W", "110°W", "105°W", "100°W", "95°W"])
    ax.set_yticklabels(["30°N", "35°N", "40°N", "45°N"])
    ax.set_xlabel("Longitude", fontsize=10, labelpad=10)
    ax.set_ylabel("Latitude", fontsize=10, labelpad=10)
    ax.tick_params(labelsize=9, length=0)
    ax.grid(color="#dbe3e9", alpha=.4, linewidth=.45, zorder=0)
    cb = fig.colorbar(image, ax=ax, fraction=.033, pad=.022, shrink=.86)
    cb.set_label("2019–2020 JJA mean ERA5 2-m temperature (°C)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8)
    handles = [Line2D([0],[0],lw=2.6,color=color,label=key)
               for key,color in (("Source","#1c72a9"),("Southwest","#009e73"),("Southern Plains","#d55e00"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.47,.075),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.075, .022,
             "Archived ERA5 temperature is shown only on the three study grids. "
             "State boundaries and city labels provide geographic context.",
             fontsize=8, color="#52646a")
    for ext in ("png", "svg"):
        path = args.output / f"paper_study_domains_era5_jja_citylabels_v2.{ext}"
        fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
