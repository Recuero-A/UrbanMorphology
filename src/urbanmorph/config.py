"""Central configuration: the sample area, grid and channel definitions.

Everything that determines the output array lives here, so a sample is fully
reproducible from (raw OSM cache + this file).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

OSM_ATTRIBUTION = (
    "Map data © OpenStreetMap contributors, available under the Open Database "
    "License (ODbL 1.0). https://www.openstreetmap.org/copyright"
)

# Channel index -> name. The order is part of the dataset contract.
CHANNELS: dict[int, str] = {0: "roads", 1: "buildings", 2: "green_spaces"}


@dataclass(frozen=True)
class AreaSpec:
    """A square sample area defined by a centre point and a metric side length.

    The square is axis-aligned in the *projected* CRS (metres), not in lat/lon.
    """

    sample_id: str
    center_lat: float
    center_lon: float
    size_m: float = 512.0
    resolution: int = 128
    # ETRS89 / UTM zone 31N: the official metric CRS for Catalonia.
    projected_crs: str = "EPSG:25831"
    # Extra download radius so features just outside the square (e.g. a road
    # whose buffer reaches inside) are present before clipping.
    download_margin_m: float = 100.0
    description: str = ""
    # Milestone 2 additions (defaults keep Milestone 1 behaviour unchanged).
    country: str = ""
    city: str = ""
    # Exact projected centre (x, y) in ``projected_crs``. When set, the sample
    # square is built from it directly, so grid-snapped tiles stay exactly on
    # the grid instead of drifting by a lat/lon round trip.
    projected_center: tuple[float, float] | None = None
    raw_subdir: str = ""  # raw cache under data/raw/<raw_subdir>/<sample_id>/

    @property
    def pixel_size_m(self) -> float:
        return self.size_m / self.resolution

    @property
    def download_dist_m(self) -> float:
        # Half-diagonal of the square covers it under any small rotation
        # between the lat/lon box and the UTM grid, plus a margin.
        return self.size_m / 2 * 2**0.5 + self.download_margin_m

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR / self.raw_subdir / self.sample_id

    def to_dict(self) -> dict:
        return asdict(self)


# First sample: central Eixample (Dreta de l'Eixample), around Pau Claris /
# Consell de Cent. Fixed centre + fixed size => reproducible query.
BARCELONA_EIXAMPLE_001 = AreaSpec(
    sample_id="barcelona_eixample_001",
    center_lat=41.3910,
    center_lon=2.1650,
    description="Barcelona, Dreta de l'Eixample (Cerdà grid), 512 m x 512 m",
)


@dataclass(frozen=True)
class RoadWidthRules:
    """Assumed carriageway widths (metres) per OSM `highway` class.

    These are documented approximations, NOT measured widths. They are used
    only when a way has no usable `width` or `lanes` tag.
    """

    default_by_class: dict[str, float] = field(
        default_factory=lambda: {
            "motorway": 20.0,
            "trunk": 16.0,
            "primary": 14.0,
            "secondary": 12.0,
            "tertiary": 10.0,
            "unclassified": 7.0,
            "residential": 8.0,
            "living_street": 6.0,
            "pedestrian": 6.0,
            "link": 6.0,  # any *_link ramp
        }
    )
    fallback_m: float = 7.0
    lane_width_m: float = 3.25
    # A tagged width outside this range is treated as a tagging error.
    plausible_range_m: tuple[float, float] = (2.0, 60.0)


ROAD_WIDTH_RULES = RoadWidthRules()

# Which OSM ways count as "roads" for channel 0. Sidewalks, footways, cycleways,
# service roads and paths are deliberately excluded: Barcelona maps sidewalks
# as separate ways, and including them would double-count street space.
ROAD_HIGHWAY_CLASSES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|pedestrian"
)

# Roads mapped as areas (closed ways with area=yes, e.g. pedestrian squares).
# Only polygon results are kept, so ordinary linear ways are ignored here.
ROAD_AREA_TAGS = {"highway": [c for c in ROAD_HIGHWAY_CLASSES.split("|")]}

BUILDING_TAGS = {"building": True}

# "Green space" = vegetated open space, regardless of public access. Not every
# green area is tagged leisure=park, so several tag families are queried.
GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve", "village_green"],
    "landuse": ["grass", "forest", "meadow", "recreation_ground", "village_green"],
    "natural": ["wood", "scrub", "grassland", "heath"],
}

# Milestone 2 (multi-city dataset) locations.
M2_DIR = PROCESSED_DIR / "m2"
M2_OUTPUTS = PROJECT_ROOT / "outputs" / "m2"
