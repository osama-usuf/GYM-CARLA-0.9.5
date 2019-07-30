# CARLA 0.9.5 - OpenAI Gym Implementation

## Main Environment:
`gym_carla/envs/carla_env.py`

## API-Client Implementation:
`gym_carla/envs/client.py`

## RoadRunner Map Files:
`RoadRunnerFiles/`

## Default GYM Baselines:
`gym_baselines/`

## Tested with:
- CARLA 0.9.5
- UE4.22.3
- OpenAI GYM 0.13.0
- Python 3.5

## Installation

```
cd gym-carla
pip install -e .
```

Add the following lines to `~/.bashrc`:

```
export UE4_ROOT=~/UnrealEngine_4.22
```

## Usage

For training:
```
python3 -m baselines.run --alg=ddpg --env=CarlaEnv-v0 --num_timesteps=1e9 --res=80x80 --save_path=./models/carla_10M_ddpg
```

For running trained model:
```
python3 -m baselines.run --alg=ddpg --env=CarlaEnv-v0 --num_timesteps=0 --res=80x80 --load_path=./models/carla_10M_ddpg --play
```
