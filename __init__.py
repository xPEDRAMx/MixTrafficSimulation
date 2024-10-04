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

    # exit_env.py
    register(
        id="exit-v0",
        entry_point="MixTrafficSimulation.envs.exit_env:ExitEnv",
    )

    # highway_env.py
    register(
        id="highway-v0",
        entry_point="MixTrafficSimulation.envs.highway_env:HighwayEnv",
    )

    register(
        id="highway-fast-v0",
        entry_point="MixTrafficSimulation.envs.highway_env:HighwayEnvFast",
    )

    # intersection_env.py
    register(
        id="intersection-v0",
        entry_point="MixTrafficSimulation.envs.intersection_env:IntersectionEnv",
    )

    register(
        id="intersection-v1",
        entry_point="MixTrafficSimulation.envs.intersection_env:ContinuousIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v0",
        entry_point="MixTrafficSimulation.envs.intersection_env:MultiAgentIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="MixTrafficSimulation.envs.intersection_env:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    # lane_keeping_env.py
    register(
        id="lane-keeping-v0",
        entry_point="MixTrafficSimulation.envs.lane_keeping_env:LaneKeepingEnv",
        max_episode_steps=200,
    )

    # merge_env.py
    register(
        id="merge-v0",
        entry_point="MixTrafficSimulation.envs.merge_env:MergeEnv",
    )

    # parking_env.py
    register(
        id="parking-v0",
        entry_point="MixTrafficSimulation.envs.parking_env:ParkingEnv",
    )

    register(
        id="parking-ActionRepeat-v0",
        entry_point="MixTrafficSimulation.envs.parking_env:ParkingEnvActionRepeat",
    )

    register(
        id="parking-parked-v0",
        entry_point="MixTrafficSimulation.envs.parking_env:ParkingEnvParkedVehicles",
    )

    # racetrack_env.py
    register(
        id="racetrack-v0",
        entry_point="MixTrafficSimulation.envs.racetrack_env:RacetrackEnv",
    )

    # roundabout_env.py
    register(
        id="roundabout-v0",
        entry_point="MixTrafficSimulation.envs.roundabout_env:RoundaboutEnv",
    )

    # two_way_env.py
    register(
        id="two-way-v0",
        entry_point="MixTrafficSimulation.envs.two_way_env:TwoWayEnv",
        max_episode_steps=15,
    )

    # u_turn_env.py
    register(id="u-turn-v0", entry_point="MixTrafficSimulation.envs.u_turn_env:UTurnEnv")

    register(
        id="midblock",
        entry_point="MixTrafficSimulation.envs.midblock_env:MidblockEnv",
    )

    register(
        id="plaza",
        entry_point="MixTrafficSimulation.envs.plaza_env:PlazaEnv",
    )


_register_highway_envs()
