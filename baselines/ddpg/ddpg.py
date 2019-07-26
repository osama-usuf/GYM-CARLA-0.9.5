import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U

from baselines import logger
import numpy as np

try:
	from mpi4py import MPI
except ImportError:
	MPI = None

def learn(network, env, 
		  seed=None,
		  total_timesteps=None,
		  nb_epochs=None, # with default settings, perform 1M steps total
		  nb_epoch_cycles=10,
		  nb_rollout_steps=20000,
		  reward_scale=1.0,
		  render=False,
		  render_eval=False,
		  noise_type='adaptive-param_0.2',
		  normalize_returns=False,
		  normalize_observations=True,
		  critic_l2_reg=10e-2,
		  actor_lr=10e-4,
		  critic_lr=10e-3,
		  popart=False,
		  gamma=0.99,
		  clip_norm=None,
		  nb_train_steps=100, # per epoch cycle and MPI worker,
		  nb_eval_steps=100,
		  batch_size=64, # per MPI worker
		  tau=0.001,
		  eval_env=None,
		  param_noise_adaption_interval=50,
		  load_path=None,
		  **network_kwargs):

	set_global_seeds(seed)
	if total_timesteps is not None:
		assert nb_epochs is None
		nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
	else:
		nb_epochs = 500
	if MPI is not None:
		rank = MPI.COMM_WORLD.Get_rank()
	else:
		rank = 0

	nb_actions = env.action_space.shape[-1]
	assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

	memory = Memory(limit=int(1e5), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
	critic = Critic(network=network, **network_kwargs)
	actor = Actor(nb_actions, network=network, **network_kwargs)

	action_noise = None
	param_noise = None
	if noise_type is not None:
		for current_noise_type in noise_type.split(','):
			current_noise_type = current_noise_type.strip()
			if current_noise_type == 'none':
				pass
			elif 'adaptive-param' in current_noise_type:
				_, stddev = current_noise_type.split('_')
				param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
			elif 'normal' in current_noise_type:
				_, stddev = current_noise_type.split('_')
				action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
			elif 'ou' in current_noise_type:
				_, stddev = current_noise_type.split('_')
				action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
			else:
				raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

	max_action = env.action_space.high
	logger.info('scaling actions by {} before executing in env'.format(max_action))

	agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
		gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
		batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
		actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
		reward_scale=reward_scale)
	
	logger.info('Using agent with the following configuration:')
	logger.info(str(agent.__dict__.items()))

	eval_episode_rewards_history = deque(maxlen=100)
	episode_rewards_history = deque(maxlen=100)
	sess = U.get_session()
	
	# Prepare everything.
	agent.initialize(sess)
	if load_path is not None:
		agent.load(load_path)
	sess.graph.finalize()
	agent.reset()
	if eval_env is not None:
		eval_obs = eval_env.reset()
	nenvs = 1 # Only simulating one environment - not parallel

	episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
	episode_step = np.zeros(nenvs, dtype = int) # vector
	episodes = 0 #scalar
	t = 0 # steps, scalar
	total_t = 0 # total steps, scalar
	epoch = 0
	start_time = time.time()

	epoch_episode_rewards = []
	epoch_episode_steps = []
	epoch_actions = []
	epoch_qs = []
	epoch_episodes = 0
	done = False
	for epoch in range(nb_epochs):
		print('\n======= EPOCH '+str(epoch+1)+'/'+str(nb_epochs)+' =======\n')
		obs = env.reset()
		for cycle in range(nb_epoch_cycles):
			print('\n======= CYCLE: '+str(cycle+1)+'/'+str(nb_epoch_cycles)+' =======\n')
			# Perform rollouts.
			if nenvs > 1:
				# if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
				# of the environments, so resetting here instead
				agent.reset()
			for t_rollout in range(nb_rollout_steps):
				# Predict next action.
				action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)
				if action[0] < 0:
					print('\nACTION',action)
				# Execute next action.
				if rank == 0 and render:
					env.render()
				# max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
				new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
				# note these outputs are batched from vecenv

				t += 1
				total_t += 1
				episode_reward += r
				episode_step += 1
				if (t % 1000 == 0):
					print("step#:", t, "reward:", round(r, 4), "total_reward:", round(episode_reward[0], 4), "done:", done)
				# Book-keeping.
				epoch_actions.append(action)
				epoch_qs.append(q)
				agent.store_transition(obs, action, r, new_obs, done) #the batched data will be unrolled in memory.py's append.
				obs = new_obs
				
				if done:
					print('\nEpisode:', epoch_episodes+1,'complete.')
					print('Total reward:',round(episode_reward[0], 4))
					epoch_episode_rewards.append(episode_reward[0])
					episode_rewards_history.append(episode_reward[0])
					epoch_episode_steps.append(episode_step[0])
					episode_reward[0] = 0.
					episode_step[0] = 0
					epoch_episodes += 1
					episodes += 1
					if nenvs == 1:
						agent.reset()
					obs = env.reset()
					t = 0
					done = False
					break

			# Train.
			print('\nTRAINING...')
			epoch_actor_losses = []
			epoch_critic_losses = []
			epoch_adaptive_distances = []
			for t_train in range(nb_train_steps):
				# Adapt param noise, if necessary.
				if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
					distance = agent.adapt_param_noise()
					epoch_adaptive_distances.append(distance)
				cl, al = agent.train()
				epoch_critic_losses.append(cl)
				epoch_actor_losses.append(al)
				agent.update_target_net()

			# Evaluate.

			eval_episode_rewards = []
			eval_qs = []
			if eval_env is not None:
				print('\nEVALUATING...\n')
				nenvs_eval = 1
				eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
				for t_rollout in range(nb_eval_steps):
					eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
					eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
					eval_episode_reward += eval_r
					eval_qs.append(eval_q)
					if eval_done:
						eval_episode_rewards.append(eval_episode_reward[0])
						eval_episode_rewards_history.append(eval_episode_reward[0])
						eval_episode_reward[0] = 0.0

		if MPI is not None:
			mpi_size = MPI.COMM_WORLD.Get_size()
		else:
			mpi_size = 1

		# Log stats.
		# XXX shouldn't call np.mean on variable length lists
		duration = time.time() - start_time
		stats = agent.get_stats()
		combined_stats = stats.copy()
		combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
		combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
		combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
		combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
		combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
		combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
		combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
		combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
		combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
		combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
		combined_stats['total/duration'] = duration
		combined_stats['total/steps_per_second'] = float(total_t) / float(duration)
		combined_stats['total/episodes'] = episodes
		combined_stats['rollout/episodes'] = epoch_episodes
		combined_stats['rollout/actions_std'] = np.std(epoch_actions)
		# Evaluation statistics.
		if eval_env is not None:
			combined_stats['eval/return'] = eval_episode_rewards
			combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
			combined_stats['eval/Q'] = eval_qs
			combined_stats['eval/episodes'] = len(eval_episode_rewards)
		def as_scalar(x):
			if isinstance(x, np.ndarray):
				assert x.size == 1
				return x[0]
			elif np.isscalar(x):
				return x
			else:
				raise ValueError('expected scalar, got %s'%x)

		combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
		if MPI is not None:
			combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

		combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

		# Total statistics.
		combined_stats['total/epochs'] = epoch + 1
		combined_stats['total/steps'] = total_t

		for key in sorted(combined_stats.keys()):
			logger.record_tabular(key, combined_stats[key])

		if rank == 0:
			logger.dump_tabular()
		logger.info('')
		logdir = logger.get_dir()
		if rank == 0 and logdir:
			if hasattr(env, 'get_state'):
				with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
					pickle.dump(env.get_state(), f)
			if eval_env and hdasattr(eval_env, 'get_state'):
				with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
					pickle.dump(eval_env.get_state(), f)


	return agent
