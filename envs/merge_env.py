from __future__ import annotations

import numpy as np

from MixTrafficSimulation import utils
from MixTrafficSimulation.envs.common.abstract import AbstractEnv
from MixTrafficSimulation.road.lane import LineType, SineLane, StraightLane
from MixTrafficSimulation.road.road import Road, RoadNetwork
from MixTrafficSimulation.vehicle.controller import ControlledVehicle
from MixTrafficSimulation.vehicle.objects import Obstacle
from MixTrafficSimulation.envs.common.graphics import EnvViewer
from MixTrafficSimulation.envs.common.graphics import FixedCameraEnvViewer

class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "simulation_frequency": 15,  # Increase simulation frequency for smoother updates
                "policy_frequency": 5,
                "real_time_rendering": True,  # Render in real-time
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the simulation has run for a sufficient number of steps."""

        # Optionally set a time limit for the simulation, e.g., 1000 steps
        #if self.sim_time > 1000:
        #    return True  # End simulation after 1000 steps

        return self.vehicle.crashed  # Only terminate if a crash occurs

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self.sim_time = 0  # Initialize the simulation time
        self.config["screen_width"] = 1200  # Adjust width as needed
        self.config["screen_height"] = 400  # Adjust height as needed
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        lanes_count = self.config.get("lanes_count", 3)  # Default to 3 lanes if not specified
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        ################################## ADDED #############################################
        if lanes_count == 2:
            y = [0 * StraightLane.DEFAULT_WIDTH, 1 * StraightLane.DEFAULT_WIDTH]
            line_type = [[c, s], [n, c]]
            line_type_merge = [[c, s], [n, s]]
        elif lanes_count == 3:
            y = [-1 * StraightLane.DEFAULT_WIDTH, 0 * StraightLane.DEFAULT_WIDTH, 1 * StraightLane.DEFAULT_WIDTH]
            line_type = [[c, s], [n, s], [n, c]]
            line_type_merge = [[c, s], [n, s], [n, s]]
        elif lanes_count == 4:
            y = [-2 * StraightLane.DEFAULT_WIDTH, -1 * StraightLane.DEFAULT_WIDTH, 0 * StraightLane.DEFAULT_WIDTH,
                 1 * StraightLane.DEFAULT_WIDTH]
            line_type = [[c, s], [n, s], [n, s], [n, c]]
            line_type_merge = [[c, s], [n, s], [n, s], [n, s]]
        elif lanes_count == 5:
            y = [-3 * StraightLane.DEFAULT_WIDTH, -2 * StraightLane.DEFAULT_WIDTH, -1 * StraightLane.DEFAULT_WIDTH,
                 0 * StraightLane.DEFAULT_WIDTH, 1 * StraightLane.DEFAULT_WIDTH]
            line_type = [[c, s], [n, s], [n, s], [n, s], [n, c]]
            line_type_merge = [[c, s], [n, s], [n, s], [n, s], [n, s]]
        else:
            raise ValueError("lanes_count must be between 2 and 5")
        ###############################################################################

        for i in range(lanes_count):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        road = self.road

        # Ego vehicle (initial vehicle on the highway)
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Initial vehicles on the highway
        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        # Initial merging vehicle
        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20
        )
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)

        self.vehicle = ego_vehicle

    def add_vehicle_periodically(self, interval=10):
        """Add vehicles periodically to both highway and merging lane."""
        if self.sim_time % interval == 0:
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

            # Add a vehicle to the highway
            highway_vehicle = other_vehicles_type(
                self.road,
                self.road.network.get_lane(("a", "b", self.np_random.integers(2))).position(0, 0),
                speed=self.np_random.uniform(28, 32)
            )
            self.road.vehicles.append(highway_vehicle)

            # Add a merging vehicle to the ramp
            merging_vehicle = other_vehicles_type(
                self.road,
                self.road.network.get_lane(("j", "k", 0)).position(110, 0),
                speed=self.np_random.uniform(20, 25)
            )
            merging_vehicle.target_speed = 30
            self.road.vehicles.append(merging_vehicle)

    ################################## added ##################################################################################
    def render(self, mode: str = "human"):
        """Override the render method to use a fixed camera."""

        # Use the custom FixedCameraEnvViewer instead of the default EnvViewer
        if self.viewer is None:
            self.viewer = FixedCameraEnvViewer(self)  # Initialize the custom viewer

        self.viewer.set_bounds(0, 700, -100, 1000)  # Adjust bounds for the merge area

        # Set the camera window position
        self.viewer.set_window_position(200, 0)

        return super().render()

    def window_position(self) -> np.ndarray:
        """Override camera to stay fixed on the merge area."""
        # Fix the position of the camera on the merge area
        return np.array([200, 0])  # Adjust coordinates to match the merge area

    def step(self, action: int):
        """The main step function that advances the simulation."""

        # Simulate one step of the environment
        result = super().step(action)  # Call the parent class step method

        # Keep track of simulation time
        self.sim_time += 1

        # Periodically add vehicles to the simulation
        self.add_vehicle_periodically(interval=10)  # Adds new vehicles every 50 steps

        return result


