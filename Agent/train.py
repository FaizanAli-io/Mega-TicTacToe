from agent import DQNAgent
from game import TicTacToe7x7

agent = DQNAgent()
env = TicTacToe7x7()

episodes = 1000
update_target_every = 50

# Load a saved model (optional)
# checkpoint_path = "saved_models/checkpoint_episode_500.pt"
# start_episode = DQNAgent.load_model(agent, checkpoint_path)
start_episode = 0

for episode in range(start_episode, episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        valid_moves = env.get_valid_moves()
        action = agent.act(state, valid_moves)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay()

    if episode % update_target_every == 0:
        agent.update_target_model()

    print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    if episode % 100 == 0:  # Save every 100 episodes
        DQNAgent.save_model(agent, episode)
