import operator
import math
import random
import operator as op

from functools import reduce
from threading import Thread
from collections import defaultdict
from typing import (
    Any, Dict, Type, Callable, Text, List, Set, Optional
)

import numpy as np
import pandas as pd
from py import process
import pygame as pg

from numpy import ndarray as Array
from numpy_ringbuffer import RingBuffer
from pygame.constants import KSCAN_J

from tunafish import AutomaticFunction


class GrowthStategy(AutomaticFunction):
    def evaluate(self, strategy: Callable, creature: 'Creature') -> float:
        pass

    def interface(self, health: float, energy: float) -> str:
        raise NotImplemented

    def const_strength(self) -> float:
        return self.context['creature'].strength

    def const_longevity(self) -> float:
        return self.context['creature'].longevity



class Light:
    def __init__(self, env, position, radius, intensity: int = 1):
        self.env = env
        self.position = np.array(position)
        self.radius = radius
        self.intensity = intensity
        self.is_attached_to_cursor = False

    def attach_to_cursor(self):
        self.is_attached_to_cursor = True


class Environment:
    def __init__(self, grid_shape: tuple, cell_size: tuple):
        self.grid_shape = np.array(grid_shape)
        self.cell_count = np.product(grid_shape)
        self.cell_size = np.array(cell_size)
        self.grid = np.empty(grid_shape, dtype=Cell)
        self.creatures = set()
        self.lights = []

    def add_light(self, position, radius, intensity=1.0) -> Light:
        light = Light(self, position, radius, intensity)
        self.lights.append(light)
        return light

    def add_creature(self, position: tuple) -> 'Creature':
        creature = Creature(self, None)
        creature.add_cell(position)
        self.creatures.add(creature)
        return creature

    def update(self):
        # calculate the net "energy received" by each
        # creature from all the lights
        energies_received = defaultdict(float)
        illuminated_cell_counts = defaultdict(int)
        cell_energies_received = defaultdict(float)

        for light in self.lights:

            if light.is_attached_to_cursor:
                light.position = np.array(
                    np.array(pg.mouse.get_pos()) / self.cell_size, dtype=int
                )

            # TODO: Use numpy query to select the neighborhood rather than this loop
            for i in range(-light.radius, light.radius):
                i += light.position[0]
                if i < 0 or i >= self.grid_shape[0]:
                    # skip cells out of x bounds
                    continue 
                for j in range(-light.radius, light.radius):
                    j += light.position[1]
                    if j < 0 or j >= self.grid_shape[1]:
                        # skip cells out of y bounds
                        continue

                    # get illuminated cell
                    cell = self.grid[i, j]
                    if cell is None:
                        continue

                    # compute "energy received" for the cell
                    dist = np.linalg.norm(cell.position - light.position)
                    if dist > 0:
                        energy_received = light.intensity / dist**2
                    else:
                        energy_received = light.intensity

                    # XXX
                    # if cell is not None:
                    #     print(light.position, cell.position, energy_received)

                    # increment the number of cells illuminated for
                    # the creature that owns the cell
                    illuminated_cell_counts[cell.creature] += 1

                    # increment net energy received for the creature
                    energies_received[cell.creature] += energy_received

                    # increment energy received for the individual cell
                    cell_energies_received[cell] += energy_received

        # set the specific energy received on each cell that received it
        # this is used for purposes of adding visual effects to the cell
        for cell, energy_received in cell_energies_received.items():
            cell.energy_received = energy_received

        # update the cumulative energy state of each creature
        for creature in self.creatures:
            creature.update(
                energy_received=energies_received[creature],
                energized_proportion=(
                    illuminated_cell_counts[creature] / len(creature.cells)
                )
            )


class Cell:
    def __init__(self, creature, position):
        self.position = np.array(position)
        self.creature = creature
        self.energy_received = 0.0
        self.color = np.zeros(3)
        self.rect = pg.Rect(
            creature.env.cell_size[0] * position[0],
            creature.env.cell_size[1] * position[1],
            creature.env.cell_size[0],
            creature.env.cell_size[1],
        )


class Creature:
    def __init__(self, env: Environment, strategy: GrowthStategy):
        self.env = env
        self.strategy = strategy
        self.cells: Set[Cell] = set()
        self.energies_received = RingBuffer(capacity=100)
        self.energies_transfered = RingBuffer(capacity=100)
        self.energies_usable = RingBuffer(capacity=100)
        self.longevity : float = 1.0
        self.strength: float = 0.1
        self.age: float = 0.0
        self.health = 1 / (1 + math.exp(self.longevity - self.strength * self.age))
        self.usable_energy: float = 0.0
        self.color = np.random.randint(25, 80, 3)
        self.rects = []

    def add_cell(self, position: tuple) -> Optional[Cell]:
        cell = None
        if self.env.grid[position] is None:
            cell = Cell(self, position)
            cell.energy_received = self.usable_energy
            self.cells.add(cell)
            self.env.grid[position] = cell
            self.rects.append(cell.rect)
        # returns none if cell couldn't be added
        return cell

    def update(
        self,
        energy_received: float,
        energized_proportion: float,
    ):
        # calculate the "metabolic health" of the creature in terms of its
        # efficiency in transfering received energy to a usable form.
        # efficiency near 0 means the creature is dead.
        efficiency = 1 / (1 + math.exp(self.longevity - self.strength * self.age))
        energy_transfered = efficiency * energy_received

        self.health = efficiency

        if efficiency < 0.001:
            self.usable_energy = 0
            return

        self.energies_received.append(energy_received)
        self.energies_transfered.append(energy_transfered)

        # a creature with fewer cells illuminated by a light source will
        # experience more of a lag time in absorption of energy, as energy must
        # spread out to those non-illuminated cells. Using an EMA simulates
        # this lag in energy levels.
        ema_period = round((1 - energized_proportion) * len(self.cells))
        usable_energy = pd.Series(self.energies_transfered).ewm(
            span=max(1, min(len(self.energies_transfered), ema_period))
        ).mean().iloc[-1]

        self.energies_usable.append(usable_energy)
        self.usable_energy = usable_energy
        self.age += 0.01

    def draw(self, screen):
        color = self.color.copy()
        for cell in self.cells:
            energy = cell.energy_received
            max_delta = 0xff - color[0]
            color[0] += int(max_delta * min(1, energy))
            cell.color = color
            screen.fill(cell.color, cell.rect)
        return self.rects


class Simulation:
    def __init__(self, screen_size: tuple, grid_shape: tuple, fps: int = 42):
        self.screen = None
        self.screen_size = np.array(screen_size)
        self.clock = pg.time.Clock()
        self.fps = fps
        self.running = False
        self.env = Environment(
            grid_shape=grid_shape,
            cell_size=np.round(screen_size / np.array(grid_shape))
        )
    
    def setup(self):
        if not pg.get_init():
            pg.init()
        self.screen = pg.display.set_mode(self.screen_size)

    def start(self):
        self.setup()
        self.running = True
        while self.running:
            self.clock.tick(self.fps)
            for event in pg.event.get():
                self.handle_event(event)
            self.update_game_state()
            self.update_display(self.draw())

    def update_game_state(self):
        self.env.update()

    def draw(self):
        # import ipdb; ipdb.set_trace()
        rects_to_update = []
        for creature in self.env.creatures:
            # TODO: only update changed cells
            rects_to_update.extend(creature.draw(self.screen))
        return rects_to_update

    def update_display(self, rects_to_update):
        pg.display.update(rects_to_update)

    def handle_event(self, event):
        if event.type == pg.QUIT:
            self.running = False


if __name__ == '__main__':
    sim = Simulation((1000, 1000), (100, 100))

    positions = set([tuple(np.random.randint(0, 99, 2)) for i in range(60)])
    for position in positions:
        sim.env.add_creature(position)

    light = sim.env.add_light(position=(45, 45), radius=50, intensity=50.0)
    light.attach_to_cursor()

    sim.start()