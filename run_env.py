import argparse
import logging
import sys
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, messagebox, ttk

import gymnasium as gym
import pandas as pd

# Allow running as `python run_env.py` from inside this directory.
project_root = Path(__file__).resolve().parent
package_parent = project_root.parent
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))

import MixTrafficSimulation  # noqa: F401 - needed to register envs
from MixTrafficSimulation import utils

logging.getLogger("gymnasium").setLevel(logging.ERROR)

AVAILABLE_ENVS = [
    "exit-v0",
    "highway-v0",
    "intersection-v1",
    "intersection-multi-agent-v1",
    "merge-v0",
    "roundabout-v0",
    "midblock",
]
CONTROL_MODES = ["NoAction (autopilot)", "Random", "Ego Manual (keyboard)"]


def run_simulation(
    env_id: str = "roundabout-v0",
    steps: int = 100,
    duration: float = 5.0,
    lanes_count: int = 3,
    max_vehicles: int = 30,
    generation_interval: float = 3.0,
    speed_limit: float = 20.0,
    render_mode: str = "human",
    enable_pedestrians: bool = False,
    control_mode: str = "NoAction (autopilot)",
) -> None:
    use_no_action = control_mode == "NoAction (autopilot)"
    use_manual = control_mode == "Ego Manual (keyboard)"

    make_config = {}
    if use_no_action:
        make_config["action"] = {"type": "NoAction"}
    elif use_manual:
        make_config["action"] = {"type": "DiscreteMetaAction"}
        make_config["manual_control"] = True

    env = gym.make(env_id, render_mode=render_mode, config=make_config)

    cfg = env.unwrapped.config
    cfg["duration"] = duration
    cfg["enable_pedestrians"] = enable_pedestrians
    cfg["other_vehicles_type"] = "MixTrafficSimulation.vehicle.behavior.IDMVehicle"
    if use_manual:
        cfg["manual_control"] = True

    # Set optional scenario params only when available.
    if "lanes_count" in cfg:
        cfg["lanes_count"] = lanes_count
    if "max_vehicles" in cfg:
        cfg["max_vehicles"] = max_vehicles
    if "vehicles_count" in cfg:
        cfg["vehicles_count"] = max_vehicles
    if "generation_interval" in cfg:
        cfg["generation_interval"] = generation_interval

    env.reset()

    if hasattr(env.unwrapped, "road") and env.unwrapped.road and env.unwrapped.road.network:
        for lane in env.unwrapped.road.network.lanes_list():
            lane.speed_limit = speed_limit

    for _ in range(steps):
        action = "NoAction" if use_no_action else env.action_space.sample()
        try:
            _, _, done, truncated, _ = env.step(action)
        except Exception:
            # Fallback for envs that do not support string NoAction.
            _, _, done, truncated, _ = env.step(env.action_space.sample())
        env.render()
        if done or truncated:
            break

    env.close()
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_info_path = output_dir / "vehicle_info.csv"
    trajectories_path = output_dir / "trajectories.csv"

    if hasattr(base_env, "save_vehicle_info_csv"):
        base_env.save_vehicle_info_csv(str(vehicle_info_path))
    if hasattr(base_env, "save_trajectories_csv"):
        base_env.save_trajectories_csv(str(trajectories_path))
        df = pd.read_csv(trajectories_path)
        utils.plot_time_x_for_each_lane_and_id(df)


def launch_ui() -> None:
    root = Tk()
    root.title("MixTrafficSimulation Launcher")
    root.geometry("520x430")

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    env_var = StringVar(value="roundabout-v0")
    steps_var = IntVar(value=100)
    duration_var = DoubleVar(value=5.0)
    lanes_var = IntVar(value=3)
    vehicles_var = IntVar(value=30)
    generation_var = DoubleVar(value=3.0)
    speed_var = DoubleVar(value=20.0)
    render_var = StringVar(value="human")
    pedestrians_var = BooleanVar(value=False)
    control_mode_var = StringVar(value="NoAction (autopilot)")

    row = 0
    ttk.Label(frame, text="Environment").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        frame, textvariable=env_var, values=AVAILABLE_ENVS, state="readonly", width=30
    ).grid(row=row, column=1, sticky="ew", pady=4)
    row += 1

    fields = [
        ("Steps", steps_var),
        ("Duration (s)", duration_var),
        ("Lanes Count", lanes_var),
        ("Max Vehicles", vehicles_var),
        ("Generation Interval", generation_var),
        ("Lane Speed Limit", speed_var),
    ]
    for label_text, var in fields:
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

    ttk.Label(frame, text="Render Mode").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        frame, textvariable=render_var, values=["rgb_array", "human"], state="readonly"
    ).grid(row=row, column=1, sticky="ew", pady=4)
    row += 1
    ttk.Label(frame, text="Control Mode").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        frame, textvariable=control_mode_var, values=CONTROL_MODES, state="readonly"
    ).grid(row=row, column=1, sticky="ew", pady=4)
    row += 1

    ttk.Checkbutton(
        frame, text="Enable Pedestrians", variable=pedestrians_var
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
    row += 1
    ttk.Label(
        frame,
        text="Manual keys: UP/DOWN for lane change, LEFT/RIGHT for slower/faster",
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
    row += 1

    status_label = ttk.Label(frame, text="Ready.")
    status_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=10)
    row += 1

    frame.columnconfigure(1, weight=1)

    def on_run() -> None:
        status_label.config(text="Simulation running... UI will pause until finished.")
        root.update_idletasks()
        try:
            run_simulation(
                env_id=env_var.get(),
                steps=steps_var.get(),
                duration=duration_var.get(),
                lanes_count=lanes_var.get(),
                max_vehicles=vehicles_var.get(),
                generation_interval=generation_var.get(),
                speed_limit=speed_var.get(),
                render_mode=render_var.get(),
                enable_pedestrians=pedestrians_var.get(),
                control_mode=control_mode_var.get(),
            )
            status_label.config(text="Done. CSV outputs written to outputs/.")
        except Exception as exc:  # pragma: no cover - runtime safety for UI path
            messagebox.showerror("Simulation error", str(exc))
            status_label.config(text=f"Failed: {exc}")

    ttk.Button(frame, text="Run Simulation", command=on_run).grid(
        row=row, column=0, columnspan=2, pady=6
    )
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MixTrafficSimulation scenarios.")
    parser.add_argument("--ui", action="store_true", help="Launch simple UI chooser.")
    parser.add_argument("--env-id", default="roundabout-v0", choices=AVAILABLE_ENVS)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--lanes-count", type=int, default=3)
    parser.add_argument("--max-vehicles", type=int, default=30)
    parser.add_argument("--generation-interval", type=float, default=3.0)
    parser.add_argument("--speed-limit", type=float, default=20.0)
    parser.add_argument("--render-mode", default="human", choices=["rgb_array", "human"])
    parser.add_argument("--enable-pedestrians", action="store_true")
    parser.add_argument(
        "--control-mode",
        default="NoAction (autopilot)",
        choices=CONTROL_MODES,
        help="Choose ego control strategy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        import gymnasium  # noqa: F401
        import pygame  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing dependency: {missing}. Install requirements first, e.g. "
            f"`pip install gymnasium pygame numpy pandas matplotlib`"
        ) from exc

    args = parse_args()
    if args.ui:
        launch_ui()
    else:
        run_simulation(
            env_id=args.env_id,
            steps=args.steps,
            duration=args.duration,
            lanes_count=args.lanes_count,
            max_vehicles=args.max_vehicles,
            generation_interval=args.generation_interval,
            speed_limit=args.speed_limit,
            render_mode=args.render_mode,
            enable_pedestrians=args.enable_pedestrians,
            control_mode=args.control_mode,
        )