from gym.envs.registration import register

register(
    id='TruckBackerUpper-v0',
    entry_point='gym_truck_backerupper.envs:TruckBackerUpperEnv',
    max_episode_steps=2000,
    reward_threshold=2100,
)
