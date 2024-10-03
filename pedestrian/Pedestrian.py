import numpy as np
import pygame


class Pedestrian:
    def __init__(self, position, destination, mass=1.0, relaxation_time=0.5):
        self.position = np.array(position)
        self.destination = np.array(destination)  # Set destination for movement
        self.velocity = np.zeros(2)  # Initial velocity
        self.mass = mass
        self.relaxation_time = relaxation_time
        self.force = np.zeros(2)

    def calculate_social_force(self):
        """ Calculate the social force using the Social Force Model. """
        # Calculate the desired velocity towards the destination
        direction = self.destination - self.position
        desired_velocity = direction / np.linalg.norm(direction)  # Normalize direction vector

        # Calculate acceleration based on the difference between desired and current velocity
        acceleration = (desired_velocity - self.velocity) / self.relaxation_time
        self.force = self.mass * acceleration
        return self.force

    def step(self, dt):
        """ Update pedestrian dynamics (force, velocity, position) similar to vehicle dynamics. """
        self.calculate_social_force()
        self.velocity += self.force * dt / self.mass
        self.position += self.velocity * dt

        # Optional: Check if the pedestrian has reached the destination
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
        # Convert world coordinates to pixel coordinates
        pix_x, pix_y = surface.pos2pix(pedestrian.position[0], pedestrian.position[1])
        # Draw the pedestrian as a small circle
        color = (255, 0, 0)  # Red for pedestrians
        radius = 3  # Adjust the size for better visibility
        # Ensure pygame compatibility
        if isinstance(surface, pygame.Surface):
            pygame.draw.circle(surface, color, (int(pix_x), int(pix_y)), radius)
        else:
            # Handle cases if surface is not a pygame.Surface (use your surface's drawing method)
            pass  # Custom handling for other surfaces

