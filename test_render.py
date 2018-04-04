import gym
import numpy as np

env = gym.make("Breakout-v0")
observation = env.reset()
done = False
point = 0
while not done:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    point += reward

print(point)
