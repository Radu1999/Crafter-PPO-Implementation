import torch
from policy import NN_Policy
import wandb
import argparse
from agent import PPOAgent
from env import Env, VectorizedEnv
import pickle
import torch.optim as optim
from pathlib import Path
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _save_stats(episodic_returns, crt_step, path, logger=None):
    episodic_returns = torch.tensor(episodic_returns)
    total_return = episodic_returns.sum().item()
    avg_return = episodic_returns.mean().item()

    if logger:
        logger.log({"eval_avg_return": avg_return, "eval_total_return": total_return, "eval_std": episodic_returns.std().item()})

    print(
        "[{:06d}] eval results: R/ep={:03.2f} Total={:03.2f} std={:03.2f}.".format(
            crt_step, avg_return, total_return, episodic_returns.std().item()
        )
    )

    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval_fn(agent, env, crt_step, opt, logger=None):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    agent.eval = True
    agent.policy.eval()
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            with torch.no_grad():
                action, _ = agent.act(obs.unsqueeze(0))
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    agent.eval = False
    _save_stats(episodic_returns, crt_step, opt.logdir, logger=logger)
    return sum(episodic_returns) / len(episodic_returns)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},64,64),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    wandb.login(key=opt.wandb_key)
    wandb.init(project="crafter")
    set_seed(opt.seed)
    
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = opt.num_envs
    env = VectorizedEnv("train", opt, num_envs)
    eval_env = Env("eval", opt)

    policy = NN_Policy(eval_env.observation_space, eval_env.action_space).to(opt.device)
    agent = PPOAgent(policy=policy, logger=wandb, device=opt.device).to(opt.device)
    optimizer = optim.Adam(agent.policy.parameters(), lr=opt.lr)


    ep_cnts = torch.zeros(num_envs, device=env.device)
    episode_rewards = torch.zeros(num_envs, device=env.device)
    episode_lengths = torch.zeros(num_envs, device=env.device)
    M = opt.rollout_size
    total_steps = 0


    obs = env.reset()
    max_reward = 0
    eval_interval = 0

    for update in range(opt.steps // (M * num_envs)):
        for step in range(M):
            total_steps += num_envs
            eval_interval += num_envs
            actions, values = agent.act(obs)
            obs, rewards, dones, infos = env.step(actions)
            episode_rewards += rewards
            episode_lengths += 1
            agent.rewards_history.append(rewards.tolist())
            agent.terminated_history.append(dones.tolist())
            for env_idx, info in enumerate(infos):
                if info['episode_done']:
                    if agent.logger:
                        agent.logger.log({
                            "episode": ep_cnts[env_idx].item(),
                            "episode_reward": episode_rewards[env_idx].item(),
                            "episode_length": episode_lengths[env_idx].item(),
                            "steps": total_steps
                        })
                        print(
                            f'Env: {env_idx}, Episode: {ep_cnts[env_idx].item() + 1}, Reward: {episode_rewards[env_idx].item()}, Length: {episode_lengths[env_idx].item()}, Steps: {total_steps}')
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    ep_cnts[env_idx] += 1

            if eval_interval >= opt.eval_interval:
               eval_interval = 0
               eval_reward = eval_fn(agent, eval_env, total_steps, opt, logger=wandb)
               if eval_reward > max_reward:
                   max_reward = eval_reward
                   torch.save(agent.policy.state_dict(), opt.logdir + "/best_model.pth")
        
        for env_idx in range(num_envs):
            agent.terminated_history[-1][env_idx] = True
        agent.update(optimizer)


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")

    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1000000,
        help="Total number of training steps.",
    )

    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        metavar="EVAL_INTERVAL",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="EVAL_EPISODES",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Learning rate",
    )

    parser.add_argument(
        "--rollout-size",
        type=int,
        default=32,
        metavar="M",
        help="Rollout size",
    )

    parser.add_argument(
        "--wandb-key",
        type=str,
        default="",
        metavar="WANDB-KEY",
        help="Wandb key",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        metavar="NUM_ENVS",
        help="Number of parallel envs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    opt = get_options()
    main(opt)