from __future__ import annotations

import numpy as np

from MixTrafficSimulation import utils
from MixTrafficSimulation.envs.common.abstract import AbstractEnv
from MixTrafficSimulation.envs.common.action import Action
from MixTrafficSimulation.road.road import Road, RoadNetwork
from MixTrafficSimulation.utils import near_split
from MixTrafficSimulation.vehicle.controller import ControlledVehicle
from MixTrafficSimulation.vehicle.kinematics import Vehicle
from MixTrafficSimulation.pedestrian.Pedestrian import Pedestrian
from MixTrafficSimulation.pedestrian.Pedestrian import PedestrianGraphics
from MixTrafficSimulation.envs.common.graphics import FixedCameraEnvViewer

Observation = np.ndarray

class MidblockEnv(AbstractEnv):
    """
    A highway driving environment.
    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes, and avoiding collisions.
    """

    def __init__(self, config=None):
        super().__init__(config)
        # Initialize the pedestrians attribute as an empty list in the constructor
        self.pedestrians = []  # Ensure pedestrians attribute exists
        self.vehicle_creation_counter = 0  # Vehicle creation counter

    def _reset(self) -> None:
        self._create_road()
        self.vehicle_creation_counter = 0  # Reset the vehicle creation counter
        self._create_vehicles(initial=True)

        # Ensure pedestrians are created if enabled in the config
        if self.config.get("enable_pedestrians", True):
            self.pedestrians = []  # Clear pedestrians list before creating new ones
            self._create_pedestrians()  # Create pedestrians if needed

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self, initial=False) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        number_to_create = 2 if not initial else self.config["vehicles_count"]
        start_position_x = 0  # Starting x position for the first vehicle
        spacing = 15  # Space each vehicle 5 units apart along the x-axis

        lanes_count = self.config.get("lanes_count")  # Get the number of lanes from the config
        lane_width = 4  # Each lane is 4 units wide

        for i in range(number_to_create):
            lane_id = i % lanes_count  # Determine the lane based on the vehicle index
            y_position = -lane_id * lane_width  # Calculate y position based on the lane

            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=lane_id,
                spacing=self.config["ego_spacing"]
            )
            # Set position with calculated y position
            vehicle.position = np.array([start_position_x + i * spacing, y_position])
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.road.vehicles.append(vehicle)

            if initial:
                self.controlled_vehicles.append(vehicle)
            #print(f"Added new vehicle at position: {vehicle.position}, speed: {vehicle.speed}, in lane: {lane_id}")

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def render(self, mode='human'):
        """Override the render method to render pedestrians and crosswalks."""
        lanes_count = self.config.get("lanes_count")
        if self.viewer is None:
            self.viewer = FixedCameraEnvViewer(self)

        # Set bounds for the camera
        self.viewer.set_bounds(-200, 200, -200, 200)
        self.viewer.set_window_position(100, lanes_count * (2) - 2)

        # Render road, vehicles, etc.
        super().render()

        if self.config.get("enable_pedestrians", True) and not self.viewer.offscreen:
            if self.pedestrians:
                PedestrianGraphics.display(self.pedestrians, self.viewer.sim_surface, offscreen=self.viewer.offscreen)

    def _create_pedestrians(self) -> None:
        """Create and place pedestrians at the defined areas around the crosswalks."""
        if not self.config.get("enable_pedestrians", True):
            return

        areas = [[100, 110, 0, 8]]  # Example area for pedestrian spawn

        for _ in range(8):
            area = areas[np.random.randint(0, len(areas))]
            pedestrian_position = np.array([
                np.random.uniform(area[0], area[1]),
                np.random.uniform(area[2], area[3])
            ])

            velocity = np.zeros(2)
            desired_velocity = np.array([1.0, 0.0])
            pedestrian = Pedestrian(pedestrian_position, velocity, desired_velocity)
            self.pedestrians.append(pedestrian)
            print(f"Created pedestrian at position {pedestrian_position}")

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.vehicle_creation_counter += 1
        print ("YES")
        if self.vehicle_creation_counter >= 10:
            self._create_vehicles()
            self.vehicle_creation_counter = 0
        return obs, reward, done, truncated, info


class HighwayEnvFast(MidblockEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

################################################################################################

