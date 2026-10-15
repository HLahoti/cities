"""cityca — cellular-automata modelling of urban growth.

Layout (each piece is a registry of swappable components):

* ``cityca.grid``     city state: layers, derived properties, city-file I/O
* ``cityca.anchors``  anchor points and their potential field Φ
* ``cityca.engine``   growth models, factors, demand laws, allocators,
                      events, simulation loop, Monte Carlo, calibration
* ``cityca.metrics``  urban-form metrics and map comparison
* ``cityca.viz``      maps, charts and animations
* ``cityca.scenario`` JSON scenarios wiring it all together
"""

from . import engine  # noqa: F401  (registers built-in models)
from .anchors import Anchor, AnchorField
from .engine import MODELS, Simulation
from .grid import CityGrid, load_city, save_city

__version__ = "0.1.0"
__all__ = ["CityGrid", "load_city", "save_city", "Anchor", "AnchorField", "Simulation", "MODELS"]
