from highway_env.envs.exit_env import ExitEnv
from highway_env.envs.highway_env import HighwayEnv, HighwayEnvFast
from highway_env.envs.intersection_env import (
    ContinuousIntersectionEnv,
    IntersectionEnv,
    MultiAgentIntersectionEnv,
)
from highway_env.envs.lane_keeping_env import LaneKeepingEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.envs.parking_env import (
    ParkingEnv,
    ParkingEnvActionRepeat,
    ParkingEnvParkedVehicles,
)
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.envs.u_turn_env import UTurnEnv
from highway_env.envs.midblock_env import MidblockEnv, HighwayEnvFast


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