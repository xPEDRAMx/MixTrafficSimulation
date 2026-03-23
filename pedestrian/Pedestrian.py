import numpy as np
import pygame

class Pedestrian:
    def __init__(
        self,
        position,
        destination,
        mass=1.0,
        relaxation_time=0.5,
        radius=0.5,
        personal_space=1.0,
        crosswalk_id=None,
    ):
        self.position = np.array(position)
        self.start_position = np.array(position)
        self.destination = np.array(destination)  # Set destination for movement
        self.velocity = np.zeros(2)  # Initial velocity
        self.mass = mass
        self.relaxation_time = relaxation_time
        self.force = np.zeros(2)
        self.radius = radius  # Size of the pedestrian (half of the diameter)
        self.personal_space = personal_space  # Distance for personal space
        self.pedestrians = []  # Initialize an empty list for pedestrian interactions
        self.vehicles = []
        self.crosswalk_id = crosswalk_id
        self.crossing_state = "approach"
        self.crosswalk_signal_allows = True
        self.curb_wait_progress = 0.2
        self.crossing_start_progress = 0.25

        # Vehicle-awareness parameters
        self.ttc_wait_threshold = 3.0
        self.ttc_emergency_threshold = 1.5
        self.vehicle_perception_range = 25.0
        self.conflict_path_width = 6.0
        self.vehicle_repulsion_gain = 2.0
        self.emergency_speedup_gain = 2.5
        self.emergency_retreat_gain = 2.0

    def set_pedestrians(self, pedestrians):
        """ Set the list of pedestrians for social force calculation. """
        self.pedestrians = pedestrians

    def set_vehicles(self, vehicles):
        """Set nearby vehicles for interaction forces and wait/go decisions."""
        self.vehicles = vehicles

    def set_crosswalk_signal(self, allows_crossing: bool):
        """Set whether the pedestrian's assigned crosswalk currently has WALK."""
        self.crosswalk_signal_allows = bool(allows_crossing)

    def _crossing_progress(self):
        path = self.destination - self.start_position
        path_norm_sq = float(np.dot(path, path))
        if path_norm_sq < 1e-6:
            return 1.0
        return float(np.clip(np.dot(self.position - self.start_position, path) / path_norm_sq, 0.0, 1.0))

    def _nearest_approaching_vehicle(self):
        nearest_vehicle = None
        nearest_ttc = float("inf")
        nearest_distance = float("inf")
        rel_dir_for_nearest = np.zeros(2)

        path = self.destination - self.start_position
        path_norm = np.linalg.norm(path)
        path_dir = path / path_norm if path_norm > 1e-6 else np.zeros(2)

        def closest_point_on_segment(point, seg_start, seg_end):
            seg = seg_end - seg_start
            seg_norm_sq = float(np.dot(seg, seg))
            if seg_norm_sq < 1e-9:
                return seg_start
            t = float(np.dot(point - seg_start, seg) / seg_norm_sq)
            t = np.clip(t, 0.0, 1.0)
            return seg_start + t * seg

        def closest_point_on_polygon(point, polygon):
            if polygon is None or len(polygon) < 2:
                return None
            best_point = None
            best_dist = float("inf")
            for idx in range(len(polygon)):
                p0 = polygon[idx]
                p1 = polygon[(idx + 1) % len(polygon)]
                cand = closest_point_on_segment(point, p0, p1)
                dist = float(np.linalg.norm(point - cand))
                if dist < best_dist:
                    best_dist = dist
                    best_point = cand
            return best_point

        for vehicle in self.vehicles:
            if getattr(vehicle, "crashed", False):
                continue

            vehicle_polygon = None
            if hasattr(vehicle, "polygon"):
                try:
                    vehicle_polygon = np.array(vehicle.polygon())
                except Exception:
                    vehicle_polygon = None

            nearest_surface_point = (
                closest_point_on_polygon(self.position, vehicle_polygon)
                if vehicle_polygon is not None
                else None
            )
            if nearest_surface_point is None:
                nearest_surface_point = np.array(vehicle.position)

            rel = self.position - nearest_surface_point
            distance = float(np.linalg.norm(rel))
            if distance < 1e-6 or distance > self.vehicle_perception_range:
                continue
            rel_dir = rel / distance

            # Keep only vehicles approaching the pedestrian.
            v_vel = np.array(getattr(vehicle, "velocity", np.zeros(2)))
            closing_speed = float(np.dot(v_vel - self.velocity, rel_dir))
            if closing_speed <= 0:
                continue

            # Approximate conflicting lane selection: keep vehicles close to crossing corridor.
            if path_norm > 1e-6:
                lateral_offset = float(
                    np.abs(
                        np.dot(
                            nearest_surface_point - self.position,
                            np.array([-path_dir[1], path_dir[0]]),
                        )
                    )
                )
                if lateral_offset > self.conflict_path_width:
                    continue

            ttc = distance / max(closing_speed, 1e-3)
            if ttc < nearest_ttc:
                nearest_ttc = ttc
                nearest_distance = distance
                nearest_vehicle = vehicle
                rel_dir_for_nearest = rel_dir

        return nearest_vehicle, nearest_ttc, nearest_distance, rel_dir_for_nearest

    def calculate_social_force(self):
        """ Calculate the social force using the Social Force Model. """
        # Calculate the desired velocity towards the destination
        direction = self.destination - self.position
        desired_velocity = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.zeros(2)

        nearest_vehicle, nearest_ttc, nearest_distance, rel_dir = self._nearest_approaching_vehicle()
        progress = self._crossing_progress()
        at_curb = self.curb_wait_progress <= progress < self.crossing_start_progress
        if at_curb:
            should_wait_for_gap = nearest_ttc < self.ttc_wait_threshold
            should_wait_for_signal = not self.crosswalk_signal_allows
            self.crossing_state = "wait" if (should_wait_for_gap or should_wait_for_signal) else "cross"
        elif progress < self.curb_wait_progress:
            # Keep approaching curb regardless of signal.
            self.crossing_state = "approach"
        elif progress < 1.0:
            self.crossing_state = "cross"

        if self.crossing_state == "wait":
            desired_velocity = np.zeros(2)

        # Calculate acceleration based on the difference between desired and current velocity
        acceleration = (desired_velocity - self.velocity) / self.relaxation_time
        self.force = self.mass * acceleration

        # Interaction forces with other pedestrians
        for other in self.pedestrians:
            if other is not self:  # Avoid self-interaction
                # Calculate distance and direction to the other pedestrian
                diff = self.position - other.position
                distance = np.linalg.norm(diff)

                # Repulsive force if within personal space
                if distance < self.personal_space:  # Interaction distance threshold
                    # Calculate a repulsive force based on the size and personal space
                    safe_distance = max(distance, 1e-3)
                    overlap = self.personal_space - safe_distance + self.radius + other.radius
                    repulsive_force = (overlap / safe_distance) * (diff / safe_distance)  # Normalize and scale
                    self.force += repulsive_force

        # Continuous vehicle repulsive force
        if nearest_vehicle is not None and np.linalg.norm(rel_dir) > 0:
            repulse_scale = self.vehicle_repulsion_gain / max(nearest_distance, 1.0)
            self.force += repulse_scale * rel_dir

        # During crossing, react strongly to sudden approaching vehicles.
        if self.crossing_state == "cross" and nearest_vehicle is not None and nearest_ttc < self.ttc_emergency_threshold:
            travel_dir = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-6 else np.zeros(2)
            self.force += self.emergency_speedup_gain * travel_dir
            self.force += self.emergency_retreat_gain * rel_dir

    def step(self, dt):
        """ Update pedestrian dynamics (force, velocity, position) similar to vehicle dynamics. """
        self.velocity += self.force * dt / self.mass
        self.position += self.velocity * dt

        # Keep state numerically stable for rendering/simulation.
        self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
        self.position = np.nan_to_num(self.position, nan=0.0, posinf=0.0, neginf=0.0)

        # Check if the pedestrian has reached the destination
        if np.linalg.norm(self.position - self.destination) < 0.5:  # Within 0.5 meters of the destination
            self.velocity = np.zeros(2)  # Stop moving if reached
            print(f"Pedestrian has reached the destination at {self.position}")

    def update_destination(self, new_destination):
        """ Update the destination of the pedestrian. """
        self.destination = np.array(new_destination)


class PedestrianGraphics:
    """A class responsible for rendering pedestrians."""

    @classmethod
    def display(cls, pedestrians, surface, offscreen=False):
        """Display pedestrians on the simulation surface."""
        # Check if pedestrians exist and the list is not empty
        if pedestrians is None or len(pedestrians) == 0:
            return  # Skip rendering if no pedestrians are present

        # Iterate through pedestrians and render each one
        for pedestrian in pedestrians:
            cls.display_pedestrian(pedestrian, surface, offscreen)

    @classmethod
    def display_pedestrian(cls, pedestrian, surface, offscreen=False):
        """Draw an individual pedestrian."""
        if not np.isfinite(pedestrian.position).all():
            return
        # Convert world coordinates to pixel coordinates
        pix_x, pix_y = surface.pos2pix(pedestrian.position[0], pedestrian.position[1])
        if not (np.isfinite(pix_x) and np.isfinite(pix_y)):
            return
        # Draw the pedestrian as a small circle
        color = (255, 0, 0)  # Red for pedestrians
        radius = 3  # Adjust the size for better visibility
        # Ensure pygame compatibility
        if isinstance(surface, pygame.Surface):
            pygame.draw.circle(surface, color, (int(pix_x), int(pix_y)), radius)
        else:
            # Handle cases if surface is not a pygame.Surface (use your surface's drawing method)
            pass  # Custom handling for other surfaces

