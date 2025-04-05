#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib

import click
import gymnasium as gym
import wandb.integration.sb3
import stable_baselines3.common.monitor

import sbx

# Dictionary mapping algorithm names to their respective classes
ALGORITHMS = {
    'TQC': {'jax': sbx.TQC, 'torch': None},
    'PPO': {'jax': sbx.PPO, 'torch': stable_baselines3.PPO},
    'SAC': {'jax': sbx.SAC, 'torch': stable_baselines3.SAC},
    'DDPG': {'jax': sbx.DDPG, 'torch': stable_baselines3.DDPG},
    'TD3': {'jax': sbx.TD3, 'torch': stable_baselines3.TD3},
    'DQN': {'jax': sbx.DQN, 'torch': stable_baselines3.DQN},
}

@click.command()
@click.option('--algorithm', type=click.Choice(list(ALGORITHMS)), default='PPO',
              help='RL algorithm to use (default: PPO)')
@click.option('--render/--no-render', default=False, help='Enable rendering (default: False)')
@click.option('--jax/--no-jax', 'use_jax', default=True,
              help='Use JAX implementation from sbx (default: True)')
@click.option('--total-timesteps', type=int, default=10_000,
              help='Total timesteps to train (default: 10000)')
@click.option('--env', type=str, default='Pendulum-v1',
              help='Environment to use (default: Pendulum-v1)')
@click.option('--log-interval', type=int, default=100)
@click.option('--wandb/--no-wandb', 'use_wandb', default=True,
              help='Enable WandB logging (default: True)')
def main(*, algorithm: str, render: bool, use_jax: bool, total_timesteps: int, env: str,
         log_interval: int, use_wandb: bool) -> None:
    learn_kwargs = {
        'total_timesteps': total_timesteps,
        'progress_bar': True,
        'log_interval': log_interval,
    }
    if use_wandb:
        wandb_run = wandb.init(
            config=learn_kwargs,
            sync_tensorboard=True,
            save_code=True,
            monitor_gym=True,
        )
        try:
            tmux_pane_guid = os.environ['TMUX_PANE_GUID']
        except KeyError:
            pass
        else:
            try:
                pathlib.Path(f'/tmp/wandb_url_{tmux_pane_guid}').write_text(wandb_run.url)
            except Exception:
                pass
        learn_kwargs['callback'] = wandb.integration.sb3.WandbCallback(
            gradient_save_freq=100,
            verbose=2,
        )
    else:
        wandb_run = None

    print('Starting the environment setup')
    render_mode = 'human' if render else None
    env = gym.make(env, render_mode=render_mode)
    env = stable_baselines3.common.monitor.Monitor(env)
    env = stable_baselines3.common.vec_env.DummyVecEnv([lambda: env])
    print(f'Environment created: {env}')

    # Select the appropriate implementation based on jax flag
    backend = 'jax' if use_jax else 'torch'
    print(f'Using {backend} backend')

    print(f'Initializing the {algorithm} model')
    model_class = ALGORITHMS[algorithm][backend]
    model: stable_baselines3.common.base_class.BaseAlgorithm = model_class(
        'MlpPolicy', env, verbose=1,
        tensorboard_log=(f'runs/{wandb_run.id}' if wandb_run else None)
    )
    print('Starting training')

    model.learn(**learn_kwargs)
    print('Training completed')

    # print('Setting up for evaluation')
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # print(f'Initial observation: {obs}')
    # for i in range(1000):
        # print(f'Step {i}')
        # action, _states = model.predict(obs, deterministic=True)
        # print(f'Action: {action}')
        # obs, reward, done, info = vec_env.step(action)
        # print(f'Reward: {reward}, Done: {done}')

    # print('Closing environment')
    # vec_env.close()

if __name__ == '__main__':
    main()