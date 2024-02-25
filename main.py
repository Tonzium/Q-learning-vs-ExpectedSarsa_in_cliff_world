import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem


from rl_glue import RLGlue
from cliffword_env import Environment
from Qagent import QLearningAgent
from ESagent import ExpectedSarsaAgent

# Agents settings
agent_init_info = {"num_actions": 4, "num_states": 48, "epsilon": 0.1, "step_size": 0.5, "discount": 1.0, "seed": 420}

SAVE_HEATMAP = 1   # 1 = save heatmap

# Create an instance of QLearningAgent without passing agent_init_info
ql_agent = QLearningAgent()
es_agent = ExpectedSarsaAgent()

# Then, initialize the agent with agent_init_info
ql_agent.agent_init(agent_init_info)
es_agent.agent_init(agent_init_info)

np.random.seed(0)

agents = {
    "Q-learning": ql_agent,
    "Expected Sarsa": es_agent
}
env = Environment()
all_reward_sums = {} # Contains sum of rewards during episode
all_state_visits = {} # Contains state visit counts during the last 10 episodes
env_info = {}
num_runs = 100 # The number of runs
num_episodes = 200 # The number of episodes in each run

for algorithm in ["Q-learning", "Expected Sarsa"]:
    all_reward_sums[algorithm] = []
    all_state_visits[algorithm] = []
    for run in tqdm(range(num_runs)):
        agent_init_info["seed"] = run
        rl_glue = RLGlue(env, agents[algorithm])
        rl_glue.rl_init(agent_init_info, env_info)

        reward_sums = []
        state_visits = np.zeros(48)
        for episode in range(num_episodes):
            if episode < num_episodes - 10:
                # Runs an episode
                rl_glue.rl_episode(10000) 
            else: 
                # Runs an episode while keeping track of visited states
                state, action = rl_glue.rl_start()
                state_visits[state] += 1
                is_terminal = False
                while not is_terminal:
                    reward, state, action, is_terminal = rl_glue.rl_step()
                    state_visits[state] += 1
                
            reward_sums.append(rl_glue.rl_return())
            
        all_reward_sums[algorithm].append(reward_sums)
        all_state_visits[algorithm].append(state_visits)

# plot results

for algorithm in ["Q-learning", "Expected Sarsa"]:
    plt.plot(np.mean(all_reward_sums[algorithm], axis=0), label=algorithm)
plt.xlabel("Episodes")
plt.ylabel("Sum of\n rewards\n during\n episode",rotation=0, labelpad=40)
plt.ylim(-100,0)
plt.legend()
plt.tight_layout()
plt.savefig(f"Plot_{num_runs}_{num_episodes}_{agent_init_info['epsilon']}_{agent_init_info['discount']}.png")
plt.clf()

### Create heatmap

if SAVE_HEATMAP == 1:

    plt.figure(figsize=(12, 6))
    for algorithm, position in [("Q-learning", 211), ("Expected Sarsa", 212)]:
        plt.subplot(position)
        average_state_visits = np.array(all_state_visits[algorithm]).mean(axis=0)
        grid_state_visits = average_state_visits.reshape((4,12))
        grid_state_visits[0,1:-1] = np.nan
        plt.pcolormesh(grid_state_visits, edgecolors='gray', linewidth=2)
        plt.title(algorithm)
        plt.axis('off')
    cm = plt.get_cmap()
    cm.set_bad('gray')

    plt.subplots_adjust(bottom=0.0, right=0.7, top=0.85)
    cax = plt.axes([0.85, 0.0, 0.075, 1.])
        
    cbar = plt.colorbar(cax=cax)
    cbar.ax.set_ylabel("Visits during\n the last 10\n episodes", rotation=0, labelpad=70)
    
    plt.savefig(f"Heatmap_{num_runs}_{num_episodes}_{agent_init_info['epsilon']}_{agent_init_info['discount']}.png")
    plt.clf()

else:
    pass