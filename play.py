from env import Env
from agent import PPOAgent
from policy import NN_Policy
import wandb
import torch


class opt:
    logdir = "crafter"
    history_length = 4
    save_video = True


if __name__ == "__main__":

    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("train", opt)
    policy = NN_Policy(env.observation_space, env.action_space).to(opt.device)
    agent = PPOAgent(policy=policy, logger=wandb, device=opt.device).to(opt.device)

    obs = env.reset()
    done = False

    agent.policy.load_state_dict(torch.load('best_model.pth', weights_only=True))
    while not done:
        with torch.no_grad():
            action, value = agent.act(obs.unsqueeze(0))
        print(action)
        obs, reward, done, info = env.step(action)
