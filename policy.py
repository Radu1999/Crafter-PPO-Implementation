import torch.nn as nn
import torch
import timm
from torchvision import transforms

def add_gaussian_noise(actions, noise_std):
    noise = np.random.normal(0, noise_std, size=actions.shape)
    return actions + noise

class NN_Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.obs_shape = obs_space

        model_name = "tf_efficientnetv2_s.in21k"
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])
        

        feature_dim = self.encoder.num_features
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_space.n),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.log_probs_history = []
        self.rewards_history = []
        self.done_history = []

    def forward(self, x):
        
        N, F, H, W, C = x.shape 
        reshaped_frames = x.permute(0, 4, 1, 2, 3).reshape(N, C, F * H, W) 
        processed_frames = torch.stack([self.preprocess(frame) for frame in reshaped_frames])
        processed_input = processed_frames.to(x.device)
        encoding = self.encoder(processed_input)

        action_probs = self.policy_head(encoding)
        values = self.value_head(encoding)

        return action_probs, values
