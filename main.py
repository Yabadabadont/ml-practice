import gym

def main():
    env = gym.make('Acrobot-v1')
    env.reset()
    while True:
        action = env.action_space.sample()
        env.step(action)
        env.render()

if __name__ == "__main__":
    main()
