from __future__ import annotations

import numpy as np

from MixTrafficSimulation import utils
from MixTrafficSimulation.road.road import RoadNetwork
from MixTrafficSimulation.vehicle.objects import StopSign, YellowSignal
import time


class TrafficSignal(StopSign):
    shared_phase_timer = 0  # Global timer for the intersection
    current_phase = "ns"  # Start with NS green, EW red
    previous_phase = "ns"  # To track the phase before the all-red phase

    ns_green_duration = 2  # Duration of north-south green phase (3 seconds)
    ew_green_duration = 2  # Duration of east-west green phase (3 seconds)
    yellow_duration = 1  # Duration of yellow phase (2 seconds)
    all_red_duration = 0.7  # Duration of all-red phase (1 second)

    def __init__(self, road: RegulatedRoad, position: tuple[float, float], orientation: str):
        super().__init__(road, position)
        self.orientation = orientation
        self.state = "red" if orientation == "ew" else "green"
        self.heading = 0 if orientation == "ns" else np.pi / 2

        # Set yellow signal orientation based on the traffic signal orientation
        yellow_signal_heading = 0 if orientation == "ns" else np.pi / 2
        self.yellow_signal = YellowSignal(road, position,
                                          yellow_signal_heading)  # Set correct heading for yellow signal

        # Add the traffic signal to the road
        road.objects.append(self)

    @classmethod
    def update_phase(cls, dt: float):
        cls.shared_phase_timer += dt
        print(f"Global phase timer: {cls.shared_phase_timer:.2f}")  # Debugging phase timer

        # Logic to handle each phase based on shared_phase_timer
        if cls.current_phase == "ns":
            if cls.shared_phase_timer >= cls.ns_green_duration:
                cls.previous_phase = cls.current_phase
                cls.current_phase = "yellow"
                cls.shared_phase_timer = 0
                print("Switching to yellow phase after NS green")
        elif cls.current_phase == "yellow":
            if cls.shared_phase_timer >= cls.yellow_duration:
                cls.current_phase = "all_red"
                cls.shared_phase_timer = 0
                print("Switching to all-red phase after yellow")
        elif cls.current_phase == "all_red":
            if cls.shared_phase_timer >= cls.all_red_duration:
                cls.shared_phase_timer = 0
                if cls.previous_phase == "ns":
                    cls.current_phase = "ew"
                    print("Switching to EW green")
                else:
                    cls.current_phase = "ns"
                    print("Switching to NS green")
        elif cls.current_phase == "ew":
            if cls.shared_phase_timer >= cls.ew_green_duration:
                cls.previous_phase = cls.current_phase
                cls.current_phase = "yellow"
                cls.shared_phase_timer = 0
                print("Switching to yellow phase after EW green")

    def update(self):
        # Set the state based on the current phase
        if self.orientation == TrafficSignal.current_phase:
            if self.state != "green":  # Only switch if not already green
                self.switch("green")
        elif self.current_phase == "yellow":
            if self.state != "yellow":  # Only switch if not already yellow
                self.switch("yellow")
        else:
            if self.state != "red":  # Only switch if not already red
                self.switch("red")

    def switch(self, new_state: str):
        print(f"Switching signal at {self.position} from {self.state} to {new_state}")
        self.state = new_state

        # Manage visibility of the yellow signal based on current state
        if self.state == "yellow":
            if self.yellow_signal not in self.road.objects:
                print(f"Adding yellow signal at {self.position} to the road")
                self.road.objects.append(self.yellow_signal)  # Add the yellow signal
        else:
            if self.yellow_signal in self.road.objects:
                print(f"Removing yellow signal at {self.position} from the road")
                self.road.objects.remove(self.yellow_signal)  # Remove the yellow signal

    def apply_state_change(self):
        # Manage the addition/removal of signals based on their state
        if self.state == "red" and self not in self.road.objects:
            print(f"Adding red signal at {self.position} to the road")
            self.road.objects.append(self)
        elif self.state == "green" and self in self.road.objects:
            print(f"Removing red signal at {self.position} from the road")
            self.road.objects.remove(self)




