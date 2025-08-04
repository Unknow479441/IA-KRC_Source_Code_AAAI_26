from functools import partial

from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2CustomEnv
from .starcraft2 import custom_scenario_registry as sc_scenarios


# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2custom"] = partial(env_fn, env=StarCraft2CustomEnv)

s_REGISTRY = {}
s_REGISTRY.update(sc_scenarios)
