from __future__ import annotations

import numpy as np

from MixTrafficSimulation import utils
from MixTrafficSimulation.envs.common.abstract import AbstractEnv
from MixTrafficSimulation.road.lane import AbstractLane, CircularLane, LineType, StraightLane, CrosswalkLane
from MixTrafficSimulation.road.regulation import RegulatedRoad
from MixTrafficSimulation.road.road import RoadNetwork
from MixTrafficSimulation.vehicle.kinematics import Vehicle
from MixTrafficSimulation.envs.common.graphics import EnvViewer
from MixTrafficSimulation.envs.common.graphics import FixedCameraEnvViewer
from MixTrafficSimulation.vehicle.objects import StopSign, YellowSignal
from MixTrafficSimulation.pedestrian.Pedestrian import Pedestrian
from MixTrafficSimulation.pedestrian.Pedestrian import PedestrianGraphics
import time

class PlazaEnv(AbstractEnv):
    

    def __init__(self, config=None, render_mode: str = None):
        self.traffic_signal_active = True  # Initialize here first
        super().__init__(config)  # Call the parent class constructor after initialization
        self.render_mode = render_mode  # Store the render mode
        self.pedestrians = []
        self.last_pedestrian_spawn_time = 0  # Initialize last spawn time
        self.pedestrian_spawn_interval = 1  # Spawn a pedestrian every second

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "duration": 13,  # [s]
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def toggle_traffic_signal(self, state: bool) -> None:
        """Enable or disable the traffic signals."""
        self.traffic_signal_active = state

        if not state:
            # Remove traffic signals from the road when turned off
            for signal in self.traffic_signals:
                if signal in self.road.objects:
                    self.road.objects.remove(signal)
        else:
            # Add traffic signals back to the road when turned on
            for signal in self.traffic_signals:
                if signal not in self.road.objects:
                    self.road.objects.append(signal)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        # Calculate the time step for this frame
        dt = 1 / self.config["simulation_frequency"]

        current_time = time.time()
        if current_time - self.last_pedestrian_spawn_time > self.pedestrian_spawn_interval:
            self._create_pedestrians()  # Call the method to create pedestrians
            self.last_pedestrian_spawn_time = current_time

        # Clear vehicles and spawn new ones if needed
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

        return obs, reward, terminated, truncated, info

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        """
        The episode is over only when a collision occurs.
        """
        # Only terminate if any vehicle has crashed.
        #return any(vehicle.crashed for vehicle in self.road.vehicles)
        #return self.vehicle.crashed  # Only terminate if a crash occurs
        return False
    
    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        """Reset the environment and create vehicles and pedestrians."""
        # Ensure pedestrians list is re-initialized before creating pedestrians
        self.pedestrians = []  # Initialize the pedestrians list
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        self._create_pedestrians()  # Create pedestrians
        self.config["screen_width"] = 800  # Adjust width as needed
        self.config["screen_height"] = 800  # Adjust height as needed



    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                (f"o{ego_id % 4}", f"ir{ego_id % 4}", 0)
            )
            destination = self.config["destination"] or "o" + str(
                self.np_random.integers(1, 4)
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )

    ########################################################### PEDS ##############################################

    def _create_pedestrians(self):
        """Create and place pedestrians at the defined areas around the crosswalks."""

        # Check if pedestrians are enabled in the configuration
        if not self.config.get("enable_pedestrians", True):
            return  # Do not create pedestrians if they are disabled

        # Define the areas for spawning pedestrians
        areas = [
            # EW Start Area
            [20,30,40,50],
            # EW End Area
            [0,10,20,30]
        ]

        # Print the locations of the spawn areas
        print("Spawn areas:")
        for idx, area in enumerate(areas):
            print(f"Area {idx + 1}: {area}")

        # Create a pedestrian for each defined area
        for _ in range(8):  # Adjust this if you want more pedestrians
            # Randomly choose an origin area
            origin_area = areas[np.random.randint(0, len(areas))]

            # Randomly choose a different destination area
            while True:
                destination_area = areas[np.random.randint(0, len(areas))]
                if destination_area != origin_area:  # Ensure origin and destination are different
                    break

            # Randomly generate a pedestrian position within the chosen origin area
            pedestrian_position = np.array([
                np.random.uniform(origin_area[0], origin_area[1]),  # Random x within origin area
                np.random.uniform(origin_area[2], origin_area[3])  # Random y within origin area
            ])

            # Randomly generate a destination position within the chosen destination area
            destination_position = np.array([
                np.random.uniform(destination_area[0], destination_area[1]),  # Random x within destination area
                np.random.uniform(destination_area[2], destination_area[3])  # Random y within destination area
            ])

            # Create pedestrian with calculated origin and destination
            pedestrian = Pedestrian(pedestrian_position, destination_position)
            self.pedestrians.append(pedestrian)  # Add pedestrians to the list
            print(
                f"Created pedestrian at position {pedestrian_position} moving towards {destination_position}")  # Debugging: Log pedestrian creation

        # Set the pedestrians list for each pedestrian
        for pedestrian in self.pedestrians:
            pedestrian.set_pedestrians(self.pedestrians)

    ########################################################### MAKE ROADS ##############################################
    

    def _make_road(self) -> None:

        location_for_the_signals = []
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = 45
        access_length = 40  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            print(f"o{corner},ir{corner}")
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10
                ),
            )
           

            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            print(f" il{(corner - 1) % 4},o{(corner - 1) % 4}")
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],)

        self.road = road
        
    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle
    
    ##################################################### added ######################################################
    def render(self, mode='human'):
        """Override the render method to render pedestrians and crosswalks."""

        # Use the custom FixedCameraEnvViewer
        if self.viewer is None:
            self.viewer = FixedCameraEnvViewer(self)

        # Set bounds for the camera
        self.viewer.set_bounds(-200, 200, -200, 200)
        self.viewer.set_window_position(0, 10)

        # Render road, vehicles, etc.
        super().render()

        # Check if pedestrians are enabled before rendering them
        if self.config.get("enable_pedestrians", True) and not self.viewer.offscreen:
            # Render pedestrians only if enabled
            if hasattr(self, 'pedestrians') and self.pedestrians:
                PedestrianGraphics.display(self.pedestrians, self.viewer.sim_surface, offscreen=self.viewer.offscreen)

####################################################################################################################

class MultiAgentIntersectionEnv(PlazaEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config


class ContinuousIntersectionEnv(PlazaEnv):
    def __init__(self, config=None, render_mode: str = None):
        # Call parent class constructor
        super().__init__(config, render_mode)
        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
            }
        )
        return config










