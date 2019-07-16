# CARLA 0.9.5 - OpenAI Gym Implementation

## Main Environment:
gym_carla/envs/carla_env.py

## API-Client Implementation:
gym_carla/envs/client.py

## RoadRunner Map Files:
RoadRunnerFiles/

## Default GYM Baselines:
gym_baselines/

## Tested with:
- CARLA 0.9.5
- UE4.22.3
- OpenAI GYM 0.13.0
- Python 3.5

## Installation

`cd gym-carla`
`pip install -e .`

Add the following lines to ~/.bashrc:
`export UE4_ROOT=~/UnrealEngine_4.22`
`PYTHONPATH="<path to repository>/gym-carla/gym_carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg":$PYTHONPATH`
`PYTHONPATH="<path to repository>/gym-carla/gym_carla/envs":$PYTHONPATH`
`export PYTHONPATH`
