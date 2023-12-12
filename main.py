import custom_environ
import random
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import matplotlib.pyplot as plt


def plot(rewards, averages, positives, negatives, show_result=False):
    print(len(rewards))
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards)
    plt.plot(averages)
    plt.plot(positives)
    plt.plot(negatives)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig("ROFL")


env = custom_environ.GridWorldEnv()
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=custom_environ.GridWorldEnv, disable_env_checking=True)
    .evaluation(evaluation_num_workers=1)
    .build()
)

for i in range(100):
    result = algo.train()
    print(i)
    # print(pretty_print(result))
print(result["hist_stats"])
rewards = result["hist_stats"]["episode_reward"]
averages = []
negatives = []
positives = []
sum = 0
count = 0
positive_count = 0
negative_count = 0
for i in rewards:
    sum += i
    count += 1
    if i <= 0:
        positive_count += 1
    else:
        negative_count += 1
    averages.append(sum / count)
    negatives.append(negative_count)
    positives.append(positive_count)

plot(result["hist_stats"]["episode_reward"], averages, positives, negatives, True)
