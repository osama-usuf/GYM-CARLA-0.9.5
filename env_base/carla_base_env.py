
"""
OpenAI Gym compatible Driving simulation environment based on Carla 0.9.5.
Assumes a server instance to be running!
"""

SERVER_PORT = 2000

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

class CarlaEnv(gym.Env):
	def __init__(self):
		self.actor_list = []
		self.client = None
		self.blueprint_library = None
		self.car = None #The car actor that we will control
		self.init_client()

	def init_client(self):
		try:
			# First of all, we need to create the client that will send the requests
			# to the simulator. Here we'll assume the simulator is accepting
			# requests in the localhost at port 2000.
			self.client = carla.Client('localhost', SERVER_PORT)
			self.client.set_timeout(2.0)
			# Once we have a client we can retrieve the world that is currently
			# running.
			world = self.client.get_world()
		except ConnectionError:
			print('Make sure server is up and running.')
			# The world contains the list blueprints that we can use for adding new
		# actors into the simulation.
		self.blueprint_library = world.get_blueprint_library()

		# Now let's filter all the blueprints of type 'vehicle' and choose one
		# at random. 
		# Change this choice to 1 vehicle passed through the cmd
		
		#print(self.blueprint_library.filter('vehicle'))
		
		bp = random.choice(self.blueprint_library.filter('vehicle'))

		#print(bp)

		# A blueprint contains the list of attributes that define a vehicle's
		# instance, we can read them and modify some of them. For instance,
		# let's randomize its color.
		if bp.has_attribute('color'):
			color = random.choice(bp.get_attribute('color').recommended_values)
			bp.set_attribute('color', color)

		# Now we need to give an initial transform to the vehicle. We choose a
		# random transform from the list of recommended spawn points of the map.
		transform = random.choice(world.get_map().get_spawn_points())

		# So let's tell the world to spawn the vehicle.
		self.car = world.spawn_actor(bp, transform)

		# It is important to note that the actors we create won't be destroyed
		# unless we call their "destroy" function. If we fail to call "destroy"
		# they will stay in the simulation even after we quit the Python script.
		# For that reason, we are storing all the actors we create so we can
		# destroy them afterwards.
		self.actor_list.append(self.car)
		print('created %s' % self.car.type_id)

		# Let's put the vehicle to drive around.
		self.car.set_autopilot(True)

		# Let's add now a "depth" camera attached to the vehicle. Note that the
		# transform we give here is now relative to the vehicle.
		camera_bp = self.blueprint_library.find('sensor.camera.depth')
		camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
		camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.car)
		self.actor_list.append(camera)
		print('created %s' % camera.type_id)

		# Now we register the function that will be called each time the sensor
		# receives an image. In this example we are saving the image to disk
		# converting the pixels to gray-scale.
		cc = carla.ColorConverter.LogarithmicDepth
		camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame_number, cc))

		# Oh wait, I don't like the location we gave to the vehicle, I'm going
		# to move it a bit forward.
		location = self.car.get_location()
		location.x += 40
		self.car.set_location(location)
		print('moved vehicle to %s' % location)



		time.sleep(5)
		self.clear()

	def reset(self):
		pass


	def reset_env(self):
		pass


	def step(self,obs):
		#print('Server version:',self.client.get_server_version())
		return 0,0,0,0

	def clear(self):
		print("Clearing created actors..")
		for actor in self.actor_list:
			actor.destroy()
		print('Cleaned')



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
	env = CarlaEnv()
	obs = env.reset()
	done = False
	t = 0
	total_reward = 0.0
	while not done:
		t += 1
		obs, reward, done, info = env.step([1.0, 0.0])  # Full throttle, zero steering angle
		total_reward += reward
		print("step#:", t, "reward:", round(reward, 4), "total_reward:", round(total_reward, 4), "done:", done)

if __name__ == "__main__":
	main()