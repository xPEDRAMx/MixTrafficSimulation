from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.pedestrian.Pedestrian import Pedestrian
from highway_env.pedestrian.Pedestrian import PedestrianGraphics
from highway_env.envs.common.graphics import FixedCameraEnvViewer

Observation = np.ndarray



class MidblockEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

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
                lane_id=lane_id,  # This should be an ID that is valid within your road system
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
            print(f"Added new vehicle at position: {vehicle.position}, speed: {vehicle.speed}, in lane: {lane_id}")


    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        # Increment the vehicle creation counter and check if it's time to add more vehicles
        self.vehicle_creation_counter += 1
        if self.vehicle_creation_counter >= 10:  # Create a new vehicle every 10 steps
            self._create_vehicles()
            self.vehicle_creation_counter = 0  # Reset the counter

        return obs, reward, done, truncated, info

    def _reset(self):
        self._create_road()
        self.vehicle_creation_counter = 0  # Reset the vehicle creation counter
        self._create_vehicles(initial=True)

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
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
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
        lanes_count = self.config.get("lanes_count")  # Get the number of lanes from the config
        # Use the custom FixedCameraEnvViewer
        if self.viewer is None:
            self.viewer = FixedCameraEnvViewer(self)

        # Set bounds for the camera
        self.viewer.set_bounds(-200, 200, -200, 200)
        self.viewer.set_window_position(100, lanes_count*(2)-2)

        # Render road, vehicles, etc.
        super().render()

        # Check if pedestrians are enabled before rendering them
        if self.config.get("enable_pedestrians", True) and not self.viewer.offscreen:
            # Render pedestrians only if enabled
            if hasattr(self, 'pedestrians') and self.pedestrians:
                PedestrianGraphics.display(self.pedestrians, self.viewer.sim_surface, offscreen=self.viewer.offscreen)

    def _create_pedestrians(self):
        """Create and place pedestrians at the defined areas around the crosswalks."""

        # Check if pedestrians are enabled in the configuration
        if not self.config.get("enable_pedestrians", True):
            return  # Do not create pedestrians if they are disabled

        # Define the areas for spawning pedestrians
        # Each area is defined as [xmin, xmax, ymin, ymax]
        areas = [
            [self.traffic_signals[0].position[0] - 12, self.traffic_signals[0].position[0] - 10,
             self.traffic_signals[1].position[1] - 2, self.traffic_signals[1].position[1] + 2], # NS Start Area

            [self.traffic_signals[2].position[0] - 12, self.traffic_signals[2].position[0] - 10,
             self.traffic_signals[3].position[1] - 2, self.traffic_signals[3].position[1] + 2],  # NS End Area

            [self.traffic_signals[4].position[0], self.traffic_signals[4].position[0],
             self.traffic_signals[5].position[1] - 16, self.traffic_signals[5].position[1] - 12],  # EW Start Area

            [self.traffic_signals[6].position[0], self.traffic_signals[6].position[0],
             self.traffic_signals[7].position[1] - 16, self.traffic_signals[7].position[1] - 12]  # EW End Area
        ]
        print (areas)

        # Print the locations of the spawn areas
        print("Spawn areas:")
        for idx, area in enumerate(areas):
            print(f"Area {idx + 1}: {area}")

        # Create a pedestrian for each defined area
        for _ in range(8):  # Adjust this if you want more pedestrians
            # Randomly choose an area
            area = areas[np.random.randint(0, len(areas))]

            # Randomly generate a pedestrian position within the chosen area
            pedestrian_position = np.array([
                np.random.uniform(area[0], area[1]),  # Random x within area
                np.random.uniform(area[2], area[3])  # Random y within area
            ])

            velocity = np.zeros(2)  # Initial velocity
            desired_velocity = np.array([1.0, 0.0])  # Desired speed towards the right
            pedestrian = Pedestrian(pedestrian_position, velocity, desired_velocity)
            self.pedestrians.append(pedestrian)  # Add pedestrians to the list
            print(f"Created pedestrian at position {pedestrian_position}")  # Debugging: Log pedestrian creation

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
