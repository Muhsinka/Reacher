The goal of this project is to solve the reacher enviroment under 200 episodes with average score of 30. The enviroment objective is to teach a double joined arm to follow the green area. The project aims to provide a deeper understanding of Deep Deterministic policy Gradient(DDPG) that consist of actor-critic methods. 

The DDPG pseudocode used to solve the enviroment is shown below.

![ddpg_algorithm](https://user-images.githubusercontent.com/43606874/52708863-43c68c80-2f9c-11e9-9001-20c619bd057d.png)

## Neural network

The DDPG Algorithm has 2 neural networks actor and critic network. the critic network calculates the state action pairs and the actor network calculates the policy state of the agent.

```
Episode 100	Average Score: 5.40
Episode 197	Average Score: 30.02
Environment solved in 197 episodes!	Average Score: 30.02
```

![download](https://user-images.githubusercontent.com/43606874/52710859-79ba3f80-2fa1-11e9-9d57-ca649fcd2487.png)


1. DDPG paper (https://arxiv.org/pdf/1509.02971.pdf)
