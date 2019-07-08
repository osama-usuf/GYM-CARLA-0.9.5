
"""
OpenAI Gym compatible Driving simulation environment based on Carla 0.9.5.
Assumes a server instance to be running!
"""

'''

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit

'''

# Find CARLA module

import glob
import os
import sys

try:
	sys.path.append(glob.glob('../dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

# Packages/Modules

import argparse
import logging
import carla
import gym
import subprocess
import signal
import traceback
import time
from datetime import datetime
import random
import client
from client import *
import pygame

class CarlaEnv(gym.Env):
	def __init__(self,args):
		self.args = args
		self.client = None
		self.world = None
		self.clock = None
		self.display = None
		self.hud = None
		self.controller = None
		self.start_client()

	def start_client(self):
		pygame.init()
		pygame.font.init()
		self.clock = pygame.time.Clock()
		try:
			self.client = carla.Client(self.args.host, self.args.port)
			self.client.set_timeout(2.0)

			self.display = pygame.display.set_mode(
				(self.args.width, self.args.height),
				pygame.HWSURFACE | pygame.DOUBLEBUF)

			self.hud = HUD(self.args.width, self.args.height)
			self.world = World(self.client.get_world(), self.hud, self.args.filter, self.args.rolename)
			self.controller = KeyboardControl(self.world, self.args.autopilot)
			self.clock.tick_busy_loop(60)
		except ConnectionError:
			print('Client couldn\'t connect properly, please retry.')

	def reset(self):
		#reset all variables here
		if (self.world):
			self.world.restart()


	def step(self,obs):
		if self.controller.parse_events(self.client, self.world, self.clock):
			return
		self.world.tick(self.clock)
		self.world.render(self.display)
		pygame.display.flip()
		return 0,0,0,0

	def __del__(self):
		if (self.world and self.world.recording_enabled):
			self.client.stop_recorder()
		if self.world is not None:
			self.world.destroy()
		pygame.quit()

def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Manual Control Client')
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='127.0.0.1',
		help='IP of the host server (default: 127.0.0.1)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'-a', '--autopilot',
		action='store_true',
		help='enable autopilot')
	argparser.add_argument(
		'--res',
		metavar='WIDTHxHEIGHT',
		default='1280x720',
		help='window resolution (default: 1280x720)')
	argparser.add_argument(
		'--filter',
		metavar='PATTERN',
		default='vehicle.*',
		help='actor filter (default: "vehicle.*")')
	argparser.add_argument(
		'--rolename',
		metavar='NAME',
		default='hero',
		help='actor role name (default: "hero")')
	args = argparser.parse_args()

	args.width, args.height = [int(x) for x in args.res.split('x')]

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	print(__doc__)

	try:
		rl_loop(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')

def rl_loop(args):
	env = CarlaEnv(args)
	obs = env.reset()
	done = False
	t = 0
	total_reward = 0.0
	while not done:
		t += 1
		obs, reward, done, info = env.step([1.0, 0.0])  # Full throttle, zero steering angle
		total_reward += reward
		#print("step#:", t, "reward:", round(reward, 4), "total_reward:", round(total_reward, 4), "done:", done)
		#break

if __name__ == "__main__":
	main()