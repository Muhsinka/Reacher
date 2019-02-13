## Introduction

The project aims to provide a deeper understanding of Deep Deterministic policy Gradient(DDPG) that consist of actor-critic methods. The goal of this project is to solve the reacher enviroment under 200 episodes with average score of 30. The enviroment objective is to teach a double joined arm to follow the green area.

The DDPG pseudocode used to solve the enviroment is shown below.

![ddpg_algorithm](https://user-images.githubusercontent.com/43606874/52708863-43c68c80-2f9c-11e9-9001-20c619bd057d.png)

## Neural network

The DDPG Algorithm has 2 neural networks actor and critic network. The critic network calculates the state action pairs and the actor network calculates the policy state of the agent. Both of the actor-crtitic networks consists of the same neural network that can be found in the Udacity Deep RL repository DDPG Algorithm.

The unimproved neural network code consist of same layers can be found in the Udacity Deep RL repository 
includes the following:

- Fully connected layer - input: 33 (state size) | output: 128
- ReLU layer - activation function
- Batch normalization
- Fully connected layer - input: 128 |  output 128
- ReLU layer - activation function
- Fully connected layer - input: 128 | output: (action size)
- Output activation layer - tanh function

Hyperparameters used in the DDPG algorithm:

- Number of training episodes: 1000
- Maximum steps per episode: 10000
- Replay buffer size: 10000
- Batch size: 128
- Gamma (discount factor): 0.99
- Tau: 1e-3
- Adam optimizer learning rate for actor and critic: 2e-4
- Weight decay: 0


## Improvements

First attemps with unimporeved code shown no signs of learning, the agent always got around 3-5 points in average.

DDPG Code

The benchmark implementation suggested in the project instruction added.

```
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
```

```
def hard_copy_weights(self, target, source):
     """ copy weights from source to target network (part of initialization)"""
     for target_param, param in zip(target.parameters(), source.parameters()):
      target_param.data.copy_(param.data)
```

The Network 

Additional 2 batch normalization layers implemented to both actor-critic networks.
```
    def __init__(self, state_size, action_size, seed=0, fc1_units=128, fc2_units=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
```

```
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
```

## Results

After the implementations the agent was able to solve the task in 197 episodes
```
Episode 100	Average Score: 5.40
Episode 197	Average Score: 30.02
Environment solved in 197 episodes!	Average Score: 30.02
```

![download](https://user-images.githubusercontent.com/43606874/52710859-79ba3f80-2fa1-11e9-9d57-ca649fcd2487.png)

## Future Improvements

The batch normalization showed significant improvemtns to the agent needs further learning.

Differents hyperparameters tunings can be applied.

D4PG algorithm in the Google Deep Mind's paper can be implemented.

TRPO algorithm in the Trust Region Policy Optimization paper can be inplemented.

## Reference
[DDPG paper](https://arxiv.org/pdf/1509.02971.pdf).

[Google DeepMind´s paper](https://openreview.net/pdf?id=SyZipzbCb).

[Trust Region Policy Optimization paper](https://arxiv.org/pdf/1502.05477.pdf).
