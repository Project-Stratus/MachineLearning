from gymnasium.envs.registration import register

register(
    id="environments/Balloon1D-v0",
    entry_point="environments.envs:Balloon1DEnv",
)

register(
    id="environments/Balloon2D-v0",
    entry_point="environments.envs:Balloon2DEnv",
)

register(
    id="environments/Balloon3D-v0",
    entry_point="environments.envs:Balloon3DEnv",
)

