from gymnasium.envs.registration import register
from environments.core.constants import *       # noqa

# outdated 1D and 2D environments
# register(
#     id="environments/Balloon1D-v0",
#     entry_point="environments.envs:Balloon1DEnv",
# )

# register(
#     id="environments/Balloon2D-v0",
#     entry_point="environments.envs:Balloon2DEnv",
# )

# All 3 dims handled in 3d env
register(
    id="environments/Balloon3D-v0",
    entry_point="environments.envs:Balloon3DEnv",
)
