import crafter
import pathlib
import torch
from collections import deque

class Env:
    def __init__(self, mode, args):
        assert mode in (
            "train",
            "eval",
        ), "`mode` argument can either be `train` or `eval`"
        self.device = args.device
        env = crafter.Env()
        self.metadata = []
      
        if mode == "train":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir),
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)

    def reset(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(64, 64, 3, device=self.device))
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0), reward, done, info
        

class VectorizedEnv:
    def __init__(self, mode: str, args, num_envs: int):
        assert mode in ("train", "eval"), "`mode` argument can either be `train` or `eval`"
        self.device = args.device
        self.num_envs = num_envs

        # Initialize multiple instances of Env
        self.envs = [Env(mode, args) for _ in range(num_envs)]

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        observations = []
        for env in self.envs:
            obs = env.reset()
            observations.append(obs)
        return torch.stack(observations)  # Shape: [num_envs, window, 64, 64, 3]

    def step(self, actions):
        next_obs, rewards, dones, infos = [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action.item())
            if done:
                obs = env.reset()
                info['episode_done'] = True
            else:
                info['episode_done'] = False

            next_obs.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            torch.stack(next_obs),    # Shape: [num_envs, window, 64, 64, 3]
            torch.tensor(rewards, device=self.device),  # Shape: [num_envs]
            torch.tensor(dones, device=self.device),    # Shape: [num_envs]
            infos
        )