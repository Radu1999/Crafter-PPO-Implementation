from env import Env
from agent import PPOAgent
from policy import NN_Policy
import wandb


class opt:
    logdir = "crafter"
    history_length = 4
    device = "cpu"
    save_video = True


env = Env("train", opt)
policy = NN_Policy(env.observation_space, env.action_space)
agent = PPOAgent(policy=policy, logger=wandb, device=opt.device)

obs = env.reset()
done = False

agent.policy.load_state_dict('best_model.pth')
while not done:
    action, value = agent.act(obs)
    print(action)
    obs, reward, done, info = env.step(action)
