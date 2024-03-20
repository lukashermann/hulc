"""'Hierarchical Universal Language Conditioned Policies implementation in pytorch
:copyright: 2022 by Oier Mees
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.0.1"
__project__ = "HULC"
__author__ = "Oier Mees"
__license__ = "MIT"
__email__ = "meeso@informatik.uni-freiburg.de"



def remove_shm_from_resource_tracker():
    """
    Monkey patch multiprocessing.resource_tracker so SharedMemory won't be tracked
    More details at: https://bugs.python.org/issue38119
    """
    # pylint: disable=protected-access, import-outside-toplevel
    # Ignore linting errors in this bug workaround hack
    from multiprocessing import resource_tracker

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


# More details at: https://bugs.python.org/issue38119
remove_shm_from_resource_tracker()

