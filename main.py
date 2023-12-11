import custom_environ
import random
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# test = custom_environ.GridWorldEnv()
#
# observation, info = test.reset()
# print("Reset Obsercation:", observation)
# print("INFO: ", info)
# observation, reward, terminated, extra, info = test.step(random.randint(0,3))
# print("First OBservation: ", observation)
# count = 1
# while not terminated:
#     observation, reward, terminated, extra, info = test.step(random.randint(0, 3))
#     count+=1
#     print("step:", count)
#     print("observation:", observation)
#     print("reward:",reward)



env = custom_environ.GridWorldEnv()
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=custom_environ.GridWorldEnv, disable_env_checking=True)
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))
