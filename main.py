import gym
import mujoco_py


# sample input
#[-0.23311697  0.5834501   0.05778984]

# sample observation
#[ 1.24112338  0.00548226 -0.00220728  0.00422701  0.03508721  0.01494323
# -0.35843209 -0.40271444 -0.55819394  0.01891018  2.13388585]



def random_hopper():
    env = gym.make('Hopper-v2')
    env.reset()
    for _ in range(100000):
        env.render()

        random_move = env.action_space.sample()

        print(random_move)

        observation, reward, done, info = env.step(random_move)

        print(observation)
        
    env.close()

def hopper_testing():
    env = gym.make('Hopper-v2')

    print(env.action_space)

    print(env.observation_space)

    env.close()
    

random_hopper()

#hopper_testing()
