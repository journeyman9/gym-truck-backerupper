import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Truck_BackerUpper-v0',
    entry_point='gym_truck_backerupper.envs:TruckBackerUpperEnv',
    timestep_limit=300,
    reward_threshold=1.0,
    nondeterministic = True,
)

