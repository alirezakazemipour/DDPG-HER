from ray.rllib.algorithms.sac import SACConfig

# config = (  # 1. Configure the algorithm,
#     PPOConfig()
#     .environment("Taxi-v3")
#     .rollouts(num_rollout_workers=2)
#     .framework("torch")
#     .training(model={"fcnet_hiddens": [64, 64]})
#     .evaluation(evaluation_num_workers=1)
# )

config = SACConfig().training(gamma=0.9, lr=0.01)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=4)

algo = config.build()  # 2. build the algorithm,
algo = config.build(env="CartPole-v1")
algo.train()