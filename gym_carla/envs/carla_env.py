"""
OpenAI Gym compatible Driving simulation environment based on Carla 0.9.5.

Codebase by: Osama Yousuf - oy02945@st.habib.edu.pk
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
from gym.spaces import Box

# Set this to the path to your Carla binary
SERVER_BINARY = os.environ.get("CARLA_SERVER", os.path.expanduser("~/Desktop/Packaged-CARLA-0.9.5/LinuxNoEditor/CarlaUE4.sh"))
assert os.path.exists(SERVER_BINARY), "CARLA_SERVER environment variable is not set properly. Please check and retry"

# The default environment configuration

ENV_CONFIG = {
	"discrete_actions": True,
	"verbose": False,
	"wait_time": 10,
	"max_steps": 19500,
	"early_terminate_on_collision": True
}

# The discrete action space
# [a , b] => a -> Throttle/Brake, b -> Direction
# : a & b -> floats, [-1,1]

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

class CarlaEnv(gym.Env):
	def __init__(self,args,config=ENV_CONFIG):
		'''
		Initiate the Carla Environment.
		'''
		# Environment variables
		self.args = args
		self.server = None
		self.client = None
		self.world = None
		self.clock = None
		self.display = None
		self.hud = None
		self.controller = None
		self.config = config

		# GYM parameters
		self.spec = lambda: None
		self.spec.id = "CarlaEnv-v0"
		self.seed = 1
		self.action_space = Box(-1.0, 1.0, shape=(2,))
		#self.action_space = Box(low=np.array([0.0,-1.0]),high=np.array([1.0,1.0]))
		self.observation_space = Box(0.0, 255.0, shape=(args.height, args.width,3))
		# RL variables
		self.num_steps = 0
		self.total_reward = 0
		self.prev_measurement = None
		self.episode_id = None
		#self.start_pos = None
		self.last_obs = None
		self.last_collision_frame = 0
		self.distance = 0.0

	def start_server(self):
		'''
		Starts the server process and blocks by wait_time to ensure proper connection
		'''
		self.server = subprocess.Popen(
			[SERVER_BINARY,"-windowed", "-ResX=1280", "-ResY=720"],
			preexec_fn=os.setsid, stdout=open(os.devnull, "w"))

		live_carla_processes.add(os.getpgid(self.server.pid))
		# Wait time for the server to start before attempting client connection
		

	def start_client(self):
		'''
		Starts the client, attempts server connection, and sets up the CARLA world.
		'''
		time.sleep(self.config["wait_time"])
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
			self.controller = KeyboardControl(self.world)
			self.clock.tick_busy_loop(60)
		except ConnectionError:
			print('Client couldn\'t connect properly, please retry.')

	def reset(self):
		if (not self.server):
			self.start_server()
			self.start_client()
		self.world.restart()
		obs = []
		while (obs == []):
			obs = self.reset_env() # blocks the client until proper connection has been established and an image has been received
		return obs
	
	def reset_env(self):
		# Reset all variables here
		self.distance = 0.0
		self.num_steps = 0
		self.total_reward = 0
		self.prev_measurement = None
		self.prev_image = None
		self.episode_id = datetime.date.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
		obs = self.world.get_image()
		self.prev_measurement = self.get_measurements()
		self.prev_measurement["control"] = {
			"steer": 0,
			"throttle": 0,
			"brake": 0,
			"reverse": False,
			"hand_brake": False,
		}
		#self.start_pos = [self.prev_measurement['x'],self.prev_measurement['y']]
		
		if (obs != None):
			return self.preprocess_img(obs)
		return []

	def get_measurements(self):
		loc, vel, wp, off_track = self.world.get_vehicle_data()
		if (self.prev_measurement):
			# if backward distance should be negative, remove norm (absolute)
			self.distance += float(np.linalg.norm([loc.x - self.prev_measurement['x'], loc.y - self.prev_measurement['y']]))

		frame, intensity = self.world.get_collision_reading() # Could be split into individual impulses based on the types of actors

		if (frame > self.last_collision_frame):
			curr_collision = intensity
			self.last_collision_frame = frame
		else:
			curr_collision = 0

		measurements = {
			"episode_id": self.episode_id,
			"step": self.num_steps,
			"wp_x": wp.transform.location.x,
			"wp_y": wp.transform.location.y,
			"wp_pitch": wp.transform.rotation.pitch,
			"wp_yaw": wp.transform.rotation.yaw,
			"wp_roll": wp.transform.rotation.roll,
			"x": loc.x,
			"y": loc.y,
			"speed": vel,
			"dist": self.distance,
			"max_steps": self.config["max_steps"],
			"collision": curr_collision,
			"off_track": off_track
		}

		# Find these intersections using LaneInvasionSensor, fix RoadRunner lanes first
		# "intersection_offroad": wp.intersection_offroad,
		# "intersection_otherlane": wp.intersection_otherlane
		return measurements

	def preprocess_img(self,image):
		array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		array = np.reshape(array, (image.height, image.width, 4))
		array = array[:, :, :3]
		array = array[:, :, ::-1]
		return array

	def step(self,action):
		# rescale since ddpg returns -ve throttle values which are useless to our environment
		print('\nBEFORE',action)
		action = self.controller.rescale(action)
		print('SCALED',action)
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
		#print('\n\nSTEPPING ACTION:',action,'\n\n')
		throttle = float(np.clip(action[0], 0, 1))
		brake = float(np.abs(np.clip(action[0], -1, 0)))
		steer = float(np.clip(action[1], -1, 1))
		reverse = False
		hand_brake = False
		if self.config["verbose"]:
			print(
				"steer", steer, "throttle", throttle, "brake", brake,
				"reverse", reverse)

		# Assumes vehicle is in autonomous mode
		self.controller.apply_autonomous_control(self.world,throttle=throttle, steer=steer, brake=brake, 
										hand_brake=hand_brake, reverse=reverse)

		# Getting observations
		image = None
		while (not image): # will only block in case of connection/computing delay
			image = self.world.get_image()
		obs = self.preprocess_img(image)

		# Getting info/measurements
		curr_measurement = self.get_measurements()
		if type(action) is np.ndarray:
			curr_measurement["action"] = [float(a) for a in action]
		else:
			curr_measurement["action"] = action

		curr_measurement["control"] = {
			"steer": steer,
			"throttle": throttle,
			"brake": brake,
			"reverse": reverse,
			"hand_brake": hand_brake,
		}

		# Getting reward
		reward = self.calculate_reward(curr_measurement)
		self.total_reward += reward
		curr_measurement["reward"] = reward
		curr_measurement["total_reward"] = self.total_reward

		# Checking whether episode should terminate
		done = (self.num_steps > self.config["max_steps"] or
				(self.config["early_terminate_on_collision"] and
				 self.check_collision(curr_measurement)))
		curr_measurement["done"] = done
		self.prev_measurement = curr_measurement
		self.num_steps += 1
		self.last_obs = obs
		return obs,reward,done,curr_measurement

	def calculate_reward(self,curr_measurement):
		
		reward = 0.0

		reward += np.clip(curr_measurement["dist"] - self.prev_measurement["dist"], -10.0, 10.0)
		# if +ve distance has been covered relative to the start point

		#if (curr_measurement["speed"] <= 1): #if car vel is idle/less than 5 km/hr
		#	reward -= 0.0125 * (1 / (curr_measurement["speed"] + 1))

		# -ve reward for continuous deceleration/idle positioning
		# curr_brake = curr_measurement["control"]["brake"]
		# if (curr_brake > 0.0):
		# 	reward -= 0.25 * curr_brake

		# Collision damage
		# reward -= .00002 * (curr_measurement["collision"] - self.prev_measurement["collision"])

		# Offlane/onlane penalty/awards

		if (curr_measurement["off_track"]):
			# since a -200 rewards means episode termination, we simply give a big -ve reward
			reward -= reward + 200
			#reward -= reward+1
		return reward


	def check_collision(self,measurement):
		return bool(measurement["collision"] > 0 or measurement["total_reward"] < -200)

	def clear_server_state(self):
		if (self.world and self.world.recording_enabled):
			self.client.stop_recorder()
		if self.world is not None:
			self.world.destroy()
		pygame.quit()
		if (self.server):
			pgid = os.getpgid(self.server.pid)
			os.killpg(pgid,signal.SIGKILL)
			live_carla_processes.remove(pgid)
			self.server = None

	def __del__(self):
		self.clear_server_state()

def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Autonomous Driving Simulator - 0.9.5')
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
		'--res',
		metavar='WIDTHxHEIGHT',
		default='800x600',
		help='window resolution (default: 800x600)')
	argparser.add_argument(
		'--filter',
		metavar='PATTERN',
		default='vehicle.dodge_charger.*',
		help='actor filter (default: "vehicle.dodge_charger.*")') #for random vehicle, use vehicle.*
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


live_carla_processes = set()  # To keep track of all the Carla processes we launch to make the cleanup easier

# Default cleanup function to be executed at program termination - cleans all instances and frees memory

def cleanup():
	print("\nKilling live carla processes", live_carla_processes)
	for pgid in live_carla_processes:
		os.killpg(pgid, signal.SIGKILL)
atexit.register(cleanup)

def rl_loop(args):
	env = CarlaEnv(args)
	for ep in range(5):
		print('\nEpisode: ',ep)
		obs = env.reset()
		done = False
		t = 0
		total_reward = 0.0
		while not done:
			t += 1
			obs, reward, done, info = env.step([0, 0])  # Full throttle, zero steering angle
			total_reward += reward
			if (t % 100 == 0):
				print("step#:", t, "reward:", round(reward, 4), "total_reward:", round(total_reward, 4), "done:", done)
		

if __name__ == "__main__":
	main()