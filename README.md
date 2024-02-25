Q-Learning vs. Expected Sarsa: A Cliff World Exploration

Introduction

This repository contains the code and results of my school project focused on comparing two prominent reinforcement learning algorithms: Q-Learning and Expected Sarsa.
The objective of this project was to analyze how each algorithm approaches the task of navigating through a "cliff world," a grid-based environment where the goal is to reach a target location without falling off a cliff.

Project Overview

The cliff world environment presents a classic problem in reinforcement learning, challenging agents to learn safe and efficient paths from a start position to a goal location.
The environment is designed with a high-risk, high-reward cliff edge that agents must navigate around to reach their goal.

In this project, I implemented and compared two algorithms:

1) Q-Learning: A form of model-free, off-policy agent.
2) Expected Sarsa: A variation of the Sarsa algorithm that, like Q-Learning, is model-free but operates on-policy.

Both algorithms were tasked with learning policies to navigate through the cliff world, with the aim of exploring their differences in terms of risk-taking and overall efficiency.

Methodology

The project involved running multiple simulations with both algorithms, using a consistent set of parameters to ensure a fair comparison. Each run consisted of numerous episodes where the agent attempted to reach the goal while minimizing the number of steps taken and avoiding the cliff.

To analyze the behavior and performance of each algorithm, I focused on two main aspects:

1) Heatmaps of State Visits: Visual representations showing the frequency of visits to each state in the environment. These heatmaps provide insight into the exploration patterns and risk-taking behavior of each algorithm.
2) Sum of rewards: The sum of rewards obtained over all episodes, serving as a measure of overall efficiency and effectiveness.

Findings

The results revealed distinct differences in the strategies adopted by Q-Learning and Expected Sarsa:

1) Q-Learning: This algorithm demonstrated a more risk-taking approach, often walking close to the cliff edge to minimize the path length. However, this strategy also led to occasional falls due to the exploratory actions, impacting its overall reliability.
2) Expected Sarsa: In contrast, Expected Sarsa opted for safer, longer paths away from the cliff. This conservative strategy almost never resulted in falling off the cliff, showcasing its reliability despite potentially higher step counts.

Conclusion

The comparison between Q-Learning and Expected Sarsa in the cliff world environment highlights a fundamental trade-off between risk and safety in reinforcement learning algorithms. While Q-Learning's risk-taking can lead to more efficient paths, it also increases the potential for costly mistakes. Expected Sarsa's conservative approach, although potentially less efficient, offers greater reliability by avoiding high-risk situations.

This project has provided valuable insights into the behavior of reinforcement learning algorithms in environments where safety and efficiency are in conflict. It underscores the importance of choosing an algorithm that aligns with the specific goals and constraints of the task at hand.
