from .grid import GridPartitioner
from .metis_cython import MetisCythonPartitioner
from .metis_wrapper import MetisPartitioner

PARTITIONERS = {
    'grid': GridPartitioner,
    'metis_cython': MetisCythonPartitioner,
    'metis_wrapper': MetisPartitioner
}