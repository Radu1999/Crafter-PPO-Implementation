import torch.nn as nn
import torch
import timm
from torchvision import transforms
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), self.sigma_init))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.full((out_features,), self.sigma_init))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init)

    def forward(self, input):
        if self.training:
            weight_eps = torch.randn_like(self.weight_sigma)
            bias_eps = torch.randn_like(self.bias_sigma) if self.bias_sigma is not None else None
            weight = self.weight_mu + self.weight_sigma * weight_eps
            bias = self.bias_mu + self.bias_sigma * bias_eps if bias_eps is not None else self.bias_mu
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


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
            nn.Linear(feature_dim, 512),
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
