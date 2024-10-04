
import logging
logging.getLogger('gymnasium').setLevel(logging.ERROR)
import pandas as pd
import gymnasium as gym
import sys
sys.path.insert(0, "C:\\git\\")
import MixTrafficSimulation
from MixTrafficSimulation import vehicle
from MixTrafficSimulation import utils
from MixTrafficSimulation.vehicle.behavior import IDMVehicle


# Initialize the environment with 'human' render mode
env = gym.make('roundabout', render_mode='rgb_array',config={'action': {'type': 'NoAction'}})

#env.unwrapped.toggle_traffic_signal(True)
#env.unwrapped.config["enable_pedestrians"] = True
env.unwrapped.config["lanes_count"] = 3
env.unwrapped.config["max_vehicles"] = 30
env.unwrapped.config["duration"] = 5
env.unwrapped.config["generation_interval"] = 3.0
env.unwrapped.config["other_vehicles_type"] = "MixTrafficSimulation.vehicle.behavior.IDMVehicle"

#TimeToCollisionObservation
#env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.IDMVehicle"
env.reset()

for lane_index, lane in enumerate(env.unwrapped.road.network.lanes_list()):
    if lane_index == 0:
        lane.speed_limit = 20.0  # Set speed limit for lane 0 (left lane)
    elif lane_index == 1:
        lane.speed_limit = 20.0  # Set speed limit for lane 1 (middle lane)
    elif lane_index == 2:
        lane.speed_limit = 20.0  # Set speed limit for lane 2 (right lane)

# Set target speed for controlled vehicles
"""
if env.unwrapped.road.vehicles:
    for controlled_vehicle in env.unwrapped.road.vehicles[:env.unwrapped.config["controlled_vehicles"]]:
        controlled_vehicle.target_speed = 20.0  # Set the new target speed in m/s
"""

# Run the simulation loop
for _ in range(100):
    # Take an action (IDLE in this case)
    #action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step("NoAction")
    
    # Render the environment (this will open a window and display the frames)
    env.render()
    
    # Exit if the environment signals that it is done
    if done:
        break

# Close the environment properly
env.close()
if hasattr(env, 'unwrapped'):
    env = env.unwrapped  # This unwraps the environment to access the base class

env.save_vehicle_info_csv("C:/Users/Pedram/Desktop/test_vehicle_info.csv")
env.save_trajectories_csv("C:/Users/Pedram/Desktop/test_trajectories.csv")

df = pd.read_csv("C:/Users/Pedram/Desktop/test_trajectories.csv")  # Update the path if necessary
utils.plot_time_x_for_each_lane_and_id(df)