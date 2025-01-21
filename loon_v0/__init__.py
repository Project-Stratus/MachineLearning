from gymnasium.envs.registration import register


register(
    id="loon_v0/LoonEnv-v0",
    entry_point="loon_v0.envs:LoonEnv",
)