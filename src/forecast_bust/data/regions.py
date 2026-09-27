"""Regional reliability cells.

The prompt-level design asks for 5 deg x 5 deg cells. The real data product we can afford
to download (WB2 IFS ENS 64x32, see docs/data_sources.md) has 5.625 deg conservative grid
boxes, so each reliability region is exactly one native box whose centre lies inside the
target domain. The code is written for general multi-point regions (see `region_members`)
so a finer grid (e.g. 1.5 deg) maps several grid points to each region without changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Descriptive area tags (coarse, for readability only; not used by any model).
_AREA_TAGS = [
    ("Arabian Sea", (5.0, 22.0), (55.0, 72.0)),
    ("Bay of Bengal", (5.0, 22.0), (80.5, 95.0)),
    ("Equatorial Indian Ocean", (-10.0, 5.0), (55.0, 100.0)),
    ("South Indian Peninsula", (8.0, 17.0), (72.0, 80.5)),
    ("Central India", (17.0, 26.0), (72.0, 86.0)),
    ("Northwest India / Pakistan", (22.0, 32.0), (60.0, 77.0)),
    ("Northeast India / Bangladesh", (22.0, 30.0), (86.0, 97.0)),
    ("Himalaya / Tibetan Plateau", (28.0, 40.0), (75.0, 105.0)),
    ("Afghanistan / Iran", (28.0, 40.0), (55.0, 75.0)),
    ("Myanmar / Indochina", (10.0, 28.0), (95.0, 106.0)),
]


def area_tag(lat: float, lon: float) -> str:
    for name, (la0, la1), (lo0, lo1) in _AREA_TAGS:
        if la0 <= lat < la1 and lo0 <= lon < lo1:
            return name
    return "Surrounding region"


@dataclass(frozen=True)
class Region:
    region_id: str
    row: int
    col: int
    lat: float
    lon: float
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]
    name: str

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id, "row": self.row, "col": self.col,
            "lat": self.lat, "lon": self.lon,
            "lat_bounds": list(self.lat_bounds), "lon_bounds": list(self.lon_bounds),
            "name": self.name,
        }


def build_regions(lats: np.ndarray, lons: np.ndarray, domain: dict) -> list[Region]:
    """One region per native grid box whose centre lies inside the target domain."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    dlat = float(np.median(np.diff(np.sort(lats))))
    dlon = float(np.median(np.diff(np.sort(lons))))
    tl = sorted(x for x in lats if domain["lat_min"] <= x <= domain["lat_max"])
    tn = sorted(x for x in lons if domain["lon_min"] <= x <= domain["lon_max"])
    regions = []
    for r, la in enumerate(tl):
        for c, lo in enumerate(tn):
            regions.append(Region(
                region_id=f"R{r:02d}{c:02d}", row=r, col=c, lat=round(la, 4), lon=round(lo, 4),
                lat_bounds=(round(la - dlat / 2, 4), round(la + dlat / 2, 4)),
                lon_bounds=(round(lo - dlon / 2, 4), round(lo + dlon / 2, 4)),
                name=area_tag(la, lo),
            ))
    return regions


def region_members(region: Region, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of grid points whose centres fall in the region's bounds."""
    li = np.where((lats >= region.lat_bounds[0]) & (lats < region.lat_bounds[1]))[0]
    oi = np.where((lons >= region.lon_bounds[0]) & (lons < region.lon_bounds[1]))[0]
    return li, oi
