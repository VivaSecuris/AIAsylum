"""Where dashboards get their copy of plotly.js.

labotomy's three dashboard builders each hard-coded
``<script src="https://cdn.plot.ly/plotly-2.26.0.min.js">``. That means every
dashboard needs internet to render and announces itself to a third-party CDN
each time one is opened -- awkward for a security product whose job is
analyzing models on isolated machines.

plotly.js ships inside the ``plotly`` Python package, so it is served from the
API instead. The dependency is pinned below plotly.py 6, which bundles
plotly.js 3.x; the dashboards were written against the 2.x API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path the API mounts plotly.js at. Kept in one place so the builders and the
# StaticFiles mount cannot disagree.
PLOTLY_URL_PATH = "/static/plotly.min.js"

# Used only when the local asset cannot be found at render time.
PLOTLY_CDN_FALLBACK = "https://cdn.plot.ly/plotly-2.26.0.min.js"


def plotly_asset_path() -> Optional[Path]:
    """Locate plotly.min.js inside the installed plotly package."""
    try:
        import plotly
    except ImportError:
        logger.warning("plotly is not installed; dashboards will fall back to the CDN")
        return None

    path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not path.exists():
        logger.warning("plotly is installed but %s is missing", path)
        return None
    return path


def plotly_script_tag() -> str:
    """The script tag to embed in a dashboard.

    Prefers the locally served copy and falls back to the CDN only when the
    package asset is missing, so a dashboard still renders on a machine where
    the extra was installed without plotly.
    """
    if plotly_asset_path() is not None:
        return f'<script src="{PLOTLY_URL_PATH}"></script>'
    return f'<script src="{PLOTLY_CDN_FALLBACK}"></script>'
