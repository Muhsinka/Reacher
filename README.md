# Enviroment
## Unity ML-Agents Reacher Enviroment
![reacher](https://user-images.githubusercontent.com/43606874/50915438-df1e8c00-1449-11e9-8f08-5f46adba8ec0.gif)

For this project, you will work with the reacher enviroment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

# Distributed Training
For this project, we will provide you with two separate versions of the Unity environment:

The first version contains a single agent.
The second version contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

# Solving the Environment
Note that your project submission need only solve one of the two versions of the environment.

## Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

## Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.

This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.

![unknown](https://user-images.githubusercontent.com/43606874/50915362-aed6ed80-1449-11e9-9a68-d7816ee9bd23.png)

Plot average scores (over all agents) with each episode

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. In the case of the plot above, the environment was solved at episode 63, since the average of the average scores from episodes 64 to 163 (inclusive) was greater than +30.
