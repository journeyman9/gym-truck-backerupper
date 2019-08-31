# gym-truck-backerupper

The objective is to make the agent back up a tractor and trailer to the loading dock within 15cm and 0.1 radians, without jackknifing. Jackknifing is considered when the (hitch) angle between the tractor and trailer is >= 90 degrees. Random tracks are spawned using a modified Dubins Curves. The environment is built using the OpenAI gym format, so the environment was designed with reinforcement learning in mind--but the method for control is left up to the user.

![truck-backer-upper-gif](https://i.imgur.com/LXmfj36.gif)

## Dependencies 
```
gym
scipy
dubins
numpy
matplotlib
python3
```

## Installation
```
$ cd gym-truck-backerupper
$ pip3 install -e .
```

## Basic methods
```
env.seed()
env.reset()
env.step()
env.close()
```

## Minimal Example
```
import gym
import gym_truck_backerupper
import numpy as np

env = gym.make('TruckBackerUpper-v0')

done = False
s = env.reset()

while not done:
    env.render()

    # Example action from LQR, replace with RL policy
    K = np.array([-24.7561, 94.6538, -7.8540])
    a = np.clip(K.dot(s), env.action_space.low, env.action_space.high)

    s_, r, done, info = env.step(a)
    s = s_
env.close()
```
## Additional Methods
A path can be manually generated using Dubins Curves by choosing the starting and ending pose. The ```manual_course(q0, qg)``` method can be used with q0 being the starting pose and qg being the ending pose as arguments. The poses consist of the x-coordinate, y-coordinate, and angle in degrees. Select the poses as if a car were to drive forward. Inputting starting and ending coordinates that do not meet the criteria of the 80x80 meters squared area will cause a while loop to continue endlessly.
```
q0 = np.array([25.0, 25.0, 225.0])
qg = np.array([-25.0, -25.0, 180.0])
env = gym.make('TruckBackerUpper-v0')
env.manual_course(q0, qg)
env.reset()
``` 
It is possible to start the trailer offset from the path using the ```manual_offset(y_IC, psi_2_IC, hitch_IC)``` method. y_IC is in meters, psi_2_IC is in degrees, and hitch_IC is also in degrees.
```
env = gym.make('TruckBackerUpper-v0')
env.manual_offset(1, 5, 20)
env.reset()
```
The velocity of the tractor is constant. The value is initally set to -2.012 meters per second, but can be altered using the following method. Driving forward is possible by selecting a positive velocity.
```
env = gym.make('TruckBackerUpper-v0')
env.manual_velocity(-5)
env.reset()
```
The trailer geometry can be altered. Specifically, the trailer wheelbase L2 and hitch length h are modifiable. The default values are L2=10.192 m and h=0.00 m. When changing h to a positive value, this alters the system to represent a trailer mounted by a ball hitch instead of a fifth wheel found on tractors with semi-trailers. 
```
env = gym.make('TruckBackerUpper-v0')
env.manual_params(L2=12.192, h=-0.228)
env.reset()
```

Note: if one needs to change more obscure parameters like the maximum time per episode, then he or she needs to unwrap the environment like the following.
```
env.gym.make('TruckBackerUpper-v0').unwrapped
env.t_final = 200.0
```

## Author
Journey McDowell

## License
This project is licensed under the MIT license - see LICENSE.md for details.

## Acknowledgements
[dubins](https://github.com/AndrewWalker/Dubins-Curves) - Dubins Curves module
