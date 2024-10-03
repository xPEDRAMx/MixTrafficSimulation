from MixTrafficSimulation.envs.exit_env import ExitEnv
from MixTrafficSimulation.envs.highway_env import HighwayEnv, HighwayEnvFast
from MixTrafficSimulation.envs.intersection_env import (
    ContinuousIntersectionEnv,
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)
from MixTrafficSimulation.envs.lane_keeping_env import LaneKeepingEnv
from MixTrafficSimulation.envs.merge_env import MergeEnv
from MixTrafficSimulation.envs.parking_env import (
    ParkingEnv,
    ParkingEnvActionRepeat,
    ParkingEnvParkedVehicles,
)
from MixTrafficSimulation.envs.racetrack_env import RacetrackEnv
from MixTrafficSimulation.envs.roundabout_env import RoundaboutEnv
from MixTrafficSimulation.envs.two_way_env import TwoWayEnv
from MixTrafficSimulation.envs.u_turn_env import UTurnEnv
from MixTrafficSimulation.envs.midblock_env import MidblockEnv, HighwayEnvFast


from gymnasium.envs.registration import register

register(
    id='midblock',  # Unique identifier for the environment
    entry_point='highway_env.envs.midblock_env:MidblockEnv',  # Path to the MidblockEnv class
)

__all__ = [
    "ExitEnv",
    "HighwayEnv",
    "HighwayEnvFast",
    "IntersectionEnv",
    "ContinuousIntersectionEnv",
    "MultiAgentIntersectionEnv",
    "LaneKeepingEnv",
    "MergeEnv",
    "ParkingEnv",
    "ParkingEnvActionRepeat",
    "ParkingEnvParkedVehicles",
    "RacetrackEnv",
    "RoundaboutEnv",
    "TwoWayEnv",
    "UTurnEnv",
    "MidblockEnv",
]
