
"""
OpenAI Gym compatible Driving simulation environment based on Carla 0.9.5.
"""

'''

	W            : throttle
	S            : brake
	AD           : steer
	Q            : toggle reverse
	Space        : hand-brake
	
	K            : toggle autonomous mode
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
import atexit
import datetime 
import random
import client
from client import *
import pygame
from gym.spaces import Box, Discrete, Tuple

# Set this to the path to your Carla binary
SERVER_BINARY = os.environ.get(
	"CARLA_SERVER", os.path.expanduser("~/Desktop/Packaged-CARLA-0.9.5/LinuxNoEditor/CarlaUE4.sh"))
assert os.path.exists(SERVER_BINARY), "CARLA_SERVER environment variable is not set properly. Please check and retry"

# Default environment configuration

ENV_CONFIG = {
	"discrete_actions": True,
	"use_image_only_observations": True,
	#"scenarios": [scenario_config["Lane_Keep_Town2"]],
	"framestack": 2,  # note: only [1, 2] currently supported
	"use_depth_camera": False,
	"early_terminate_on_collision": True,
	"verbose": False,
	"render" : True,  # Render to display if true
	"render_x_res": 800,
	"render_y_res": 600,
	"x_res": 80,
	"y_res": 80,
	"seed": 1,
	"wait_time": 5
}

# Define the discrete action space
DISCRETE_ACTIONS = {
	0: [0.0, 0.0],    # Coast
	1: [0.0, -0.5],   # Turn Left
	2: [0.0, 0.5],    # Turn Right
	3: [1.0, 0.0],    # Forward
	4: [-0.5, 0.0],   # Brake
	5: [1.0, -0.5],   # Bear Left & accelerate
	6: [1.0, 0.5],    # Bear Right & accelerate
	7: [-0.5, -0.5],  # Bear Left & decelerate
	8: [-0.5, 0.5],   # Bear Right & decelerate
}

live_carla_processes = set()  # To keep track of all the Carla processes we launch to make the cleanup easier

def cleanup():
	print("Killing live carla processes", live_carla_processes)
	for pgid in live_carla_processes:
		os.killpg(pgid, signal.SIGKILL)
atexit.register(cleanup)

class CarlaEnv(gym.Env):
	def __init__(self,args,config=ENV_CONFIG):
		self.args = args
		self.server = None
		self.client = None
		self.world = None
		self.clock = None
		self.display = None
		self.hud = None
		self.controller = None
		self.config = config

		if config["discrete_actions"]:
			self.action_space = Discrete(len(DISCRETE_ACTIONS))
		else:
			self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.uint8)
		if config["use_depth_camera"]:
			image_space = Box(
				-1.0, 1.0, shape=(
					config["y_res"], config["x_res"],
					1 * config["framestack"]), dtype=np.float32)
		else:
			image_space = Box(
				0.0, 255.0, shape=(
					config["y_res"], config["x_res"],
					3 * config["framestack"]), dtype=np.float32)
		if self.config["use_image_only_observations"]:
			self.observation_space = image_space
		else:
			self.observation_space = Tuple(
				[image_space,
				 Discrete(len(COMMANDS_ENUM)),  # next_command
				 Box(-128.0, 128.0, shape=(2,), dtype=np.float32)])  # forward_speed, dist to goal

		self._spec = lambda: None
		self._spec.id = "CarlaEnv-v0"
		self._seed = ENV_CONFIG["seed"]


		self.num_steps = 0
		self.total_reward = 0
		self.prev_measurement = None
		self.prev_image = None
		self.episode_id = None
		self.measurements_file = None
		self.weather = None
		self.scenario = None
		self.start_pos = None
		self.end_pos = None
		self.start_coord = None
		self.end_coord = None
		self.last_obs = None

	def start_server(self):
		self.server = subprocess.Popen(
			[SERVER_BINARY,"-windowed", "-ResX=400", "-ResY=300"],
			preexec_fn=os.setsid, stdout=open(os.devnull, "w"))

		live_carla_processes.add(os.getpgid(self.server.pid))
		#Wait time for the server to start before attempting connection
		time.sleep(self.config["wait_time"])

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
		pygame.quit()
		self.start_server()
		self.start_client()
		self.world.restart()
		obs = None
		while (obs == None):
			obs = self.reset_env()
		return obs
	
	def reset_env(self):
		#reset all variables here
		self.num_steps = 0
		self.total_reward = 0
		self.prev_measurement = None
		self.prev_image = None
		self.episode_id = datetime.date.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
		self.measurements_file = None
		return self.world.camera_manager.image 

	def step(self,action):
		if self.controller.parse_events(self.client, self.world, self.clock):
			return
		self.world.tick(self.clock)
		self.world.render(self.display)
		pygame.display.flip()
		try:
			obs = self.step_env(action)
			return obs
		except Exception:
			print(
				"Error during step, terminating episode early",
				traceback.format_exc())
			self.clear_server_state()
			return (self.last_obs, 0.0, True, {})

	def step_env(self,action):
		throttle = float(np.clip(action[0], 0, 1))
		brake = float(np.abs(np.clip(action[0], -1, 0)))
		steer = float(np.clip(action[1], -1, 1))
		reverse = False
		hand_brake = False
		if self.config["verbose"]:
			print(
				"steer", steer, "throttle", throttle, "brake", brake,
				"reverse", reverse)

		#edit, make sure vehicle is in autonomous mode
		self.controller.apply_autonomous_control(self.world,throttle=throttle, steer=steer, brake=brake, 
										hand_brake=hand_brake, reverse=reverse)
		return 0,0,0,0


	def clear_server_state(self):
		if (self.world and self.world.recording_enabled):
			self.client.stop_recorder()
		if self.world is not None:
			self.world.destroy()
		pygame.quit()
		print('cleaning server')
		if (self.server):
			pgid = os.getpgid(self.server.pid)
			os.killpg(pgid,signal.SIGKILL)
			live_carla_processes.remove(pgid)
			self.server = None

	def __del__(self):
		self.clear_server_state()

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
	for _ in range(5):
		env = CarlaEnv(args)
		obs = env.reset()
		done = False
		t = 0
		total_reward = 0.0
		while not done:
			t += 1
			obs, reward, done, info = env.step(DISCRETE_ACTIONS[random.randint(0,len(DISCRETE_ACTIONS)-1)])  # Full throttle, zero steering angle
			total_reward += reward		
		break

if __name__ == "__main__":
	main()