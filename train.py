import gym
from agent_double import DQAgent
import numpy as np

EP = 50000

class Config:
    img_size = (210, 160, 4)
    dropout_rate = 0.75
    lr = 3e-5
    action_choices = 3
    epsilon = 1.
    epsilon_min = 0.1
    epsilon_decay = 0.999
    gamma = 0.95

def main():
    env = gym.make('Breakout-v0')

    # input dimension (210,  160, 4)
    state_space = env.observation_space.shape[:2] + (4,)
    action_size = env.action_space.n
    config = Config()
    config.action_choices = action_size
    config.img_size = state_space
    agent = DQAgent(config)
    agent.load_model('model/atariv2.h5')
    batch_size = 32

    agent.update_target_model()

    for e in range(EP):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        state = np.stack([state for _ in range(4)], axis=-1)
        next_state = state.copy()
        point = 0
        done = False
        for t in range(5000):
            action = agent.act(state)
            tmp, reward, done, _ = env.step(action)
            tmp_state = np.expand_dims(tmp, axis=0)

            # Stacking most recent 4 screens
            # Shift 3 to left (history) and append to the right
            next_state = np.roll(state, -1, axis=3)
            next_state[:,:,:,3] = tmp_state

            reward  = reward if not done else -1
            point += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state.copy()
            if done:
                print('episode {}/{}, score: {}'.format(e, EP, point))

                # Print counted frames
                print(t)
                agent.update_target_model()
                break
        agent.replay(batch_size)
        if e % 100 == 0:
            agent.save_model('model/atariv2.h5')

if __name__ == "__main__":
    main()
