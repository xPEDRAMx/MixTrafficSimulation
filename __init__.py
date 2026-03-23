import os
import sys

from gymnasium.envs.registration import register


__version__ = "1.10.1"

try:
    from farama_notifications import notifications

    if "MixTrafficSimulation" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["MixTrafficSimulation"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    from MixTrafficSimulation.envs.common.abstract import MultiAgentWrapper

    register(
        id="exit-v0",
        entry_point="MixTrafficSimulation.envs.exit_env:ExitEnv",
    )

    register(
        id="highway-v0",
        entry_point="MixTrafficSimulation.envs.highway_env:HighwayEnv",
    )

    register(
        id="intersection-v1",
        entry_point="MixTrafficSimulation.envs.intersection_env:ContinuousIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="MixTrafficSimulation.envs.intersection_env:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    register(
        id="merge-v0",
        entry_point="MixTrafficSimulation.envs.merge_env:MergeEnv",
    )

    register(
        id="roundabout-v0",
        entry_point="MixTrafficSimulation.envs.roundabout_env:RoundaboutEnv",
    )

    register(
        id="midblock",
        entry_point="MixTrafficSimulation.envs.midblock_env:MidblockEnv",
    )


_register_highway_envs()
