from datetime import datetime
import os
from typing import Literal
import numpy as np
import pygame
import jax
import tensorflow as tf
import pickle
from pathlib import Path
import json

from dweam import Game, GameInfo, Field
from .launch import (
    create_model, load_ema, setup_vae,
    diffusion_handle, SAMPLING_STEPS,
    PAST_CONTEXT_NOISE_STEPS
)
from .server import construct_action_vector
from .utils.train_state import TrainState

def get_available_maps():
    maps_dir = Path(__file__).parent / "maps"
    map_files = [f.stem for f in maps_dir.glob("*.pickle")]
    return map_files

class LucidGame(Game):
    class Params(Game.Params):
        denoising_steps: int = Field(default=SAMPLING_STEPS, description="Number of denoising steps")
        mouse_multiplier: float = Field(default=0.3, description="Mouse sensitivity multiplier")
        map_id: Literal[tuple(get_available_maps())] = Field(  # type: ignore
            default="635_256_0",
            description="Map ID to load",
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize model components
        self.model = create_model(
            patch_size=1, 
            hidden_size=2048,
            depth=16,
            num_heads=16,
            mlp_ratio=4.0,
            ctx_dropout_prob=0.1
        )
        ema_params = load_ema()
        self.state = TrainState.create(self.model, ema_params)
        self.state = jax.device_put(self.state)

        # Setup VAE decoder
        self.decode_latents = setup_vae()
        # Initialize diffusion handler
        self.diffusion_handler = diffusion_handle(
            self.decode_latents,
            self.params.denoising_steps,
            PAST_CONTEXT_NOISE_STEPS,
            self.params.map_id,
        )

        # Game state
        self.width = 640
        self.height = 480
        self.mouse_multiplier = 0.3
        
        self.action_states = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "jump": False,
            "sprint": False,
            "attack": False,
            "use": False
        }
        
        self.view_angles = {"yaw": 0.0, "pitch": 0.0}
        self.selected_inventory = 0

    def step(self) -> pygame.Surface:
        # Process current input state
        action_vector = self.construct_action_vector()
        
        # Generate new frame
        frame = self.generate_frame(action_vector)
        
        # Convert numpy array to pygame surface
        return self.display_frame(frame)

    def construct_action_vector(self):
        message = {
            "actionStates": self.action_states,
            "viewAngles": self.view_angles,
            "selectedInventory": self.selected_inventory
        }
        self.view_angles = {"yaw": 0.0, "pitch": 0.0}
        return construct_action_vector(message)

    def generate_frame(self, action_vector):
        try:
            return self.diffusion_handler(self.state, action_vector)
        except Exception:
            self.log.exception("Error in diffusion stream")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def display_frame(self, frame):
        # bgr -> rgb
        frame = frame[:, :, ::-1]
        return pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    def on_key_down(self, key: int) -> None:
        if key == pygame.K_w:
            self.action_states["forward"] = True
        elif key == pygame.K_s:
            self.action_states["backward"] = True
        elif key == pygame.K_a:
            self.action_states["left"] = True
        elif key == pygame.K_d:
            self.action_states["right"] = True
        elif key == pygame.K_SPACE:
            self.action_states["jump"] = True
        elif key == pygame.K_LSHIFT:
            self.action_states["sprint"] = True
        elif key == pygame.K_1:
            self.selected_inventory = 0
        elif key == pygame.K_2:
            self.selected_inventory = 1
        elif key == pygame.K_3:
            self.selected_inventory = 2

    def on_key_up(self, key: int) -> None:
        if key == pygame.K_w:
            self.action_states["forward"] = False
        elif key == pygame.K_s:
            self.action_states["backward"] = False
        elif key == pygame.K_a:
            self.action_states["left"] = False
        elif key == pygame.K_d:
            self.action_states["right"] = False
        elif key == pygame.K_SPACE:
            self.action_states["jump"] = False
        elif key == pygame.K_LSHIFT:
            self.action_states["sprint"] = False

    def on_mouse_motion(self, rel: tuple[int, int]) -> None:
        x, y = rel
        self.view_angles["yaw"] += x * self.mouse_multiplier
        self.view_angles["pitch"] += y * self.mouse_multiplier

    def on_mouse_down(self, button: int) -> None:
        if button == pygame.BUTTON_LEFT:
            self.action_states["attack"] = True
        elif button == pygame.BUTTON_RIGHT:
            self.action_states["use"] = True

    def on_mouse_up(self, button: int) -> None:
        if button == pygame.BUTTON_LEFT:
            self.action_states["attack"] = False
        elif button == pygame.BUTTON_RIGHT:
            self.action_states["use"] = False

    def stop(self) -> None:
        """Clean up GPU resources when stopping the game"""
        super().stop()
        
        # Force JAX to clear its cache
        jax.clear_caches()
        
        # TODO deload the model from GPU memory

    def on_params_update(self, new_params: Params) -> None:
        """Handle parameter updates for Lucid game"""
        if self.params.mouse_multiplier != new_params.mouse_multiplier:
            self.mouse_multiplier = new_params.mouse_multiplier
            self.log.info("Updated mouse sensitivity", 
                         multiplier=new_params.mouse_multiplier)

        # Reinitialize diffusion handler if relevant params changed
        # TODO reinitialize this whenever submit is pressed?
        #  it'd be nice to give people a way to reset to the same map id
        #  but also allow people to change denoising steps without reloading
        #  maybe some UI schema for this, like a reset button?
        if (self.params.denoising_steps != new_params.denoising_steps
            or self.params.map_id != new_params.map_id):
            self.log.info("Reinitializing diffusion handler",
                         denoising_steps=new_params.denoising_steps,
                         map_id=new_params.map_id)
            self.diffusion_handler = diffusion_handle(
                self.decode_latents,
                new_params.denoising_steps,
                PAST_CONTEXT_NOISE_STEPS,
                new_params.map_id,
            )
            self.log.info("Reinitialized diffusion handler",
                         denoising_steps=new_params.denoising_steps)
        
        super().on_params_update(new_params)
