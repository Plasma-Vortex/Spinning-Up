import tensorflow as tf
from tensorflow import keras
import gym

class VPG:
    def __init__(self, env_name, architecture):
        self.env = gym.make(env_name)
        self.output_dims = self.env.action_space.n
        
        self.policy = tf.keras.Sequential()
        for size in architecture:
            self.policy.add(tf.keras.layers.Dense(size, activation=tf.nn.sigmoid))
        self.policy.add(tf.keras.layers.Dense(self.output_dims, activation=None))
        self.optimizer = tf.keras.optimizers.Adam()
        
    def train(self, num_epochs, episodes_per_epoch):
        for epoch in range(num_epochs):
            self.all_states = []
            self.all_actions = []
            self.all_rewards = []
            total_reward = 0
            for _ in range(episodes_per_epoch):
                total_reward += self.generate_data()
            
            state_tensor = tf.convert_to_tensor(self.all_states)
            reward_tensor = tf.convert_to_tensor(self.all_rewards)
            with tf.GradientTape() as tape:
                log_probs = tf.nn.log_softmax(self.policy(state_tensor))
                one_hot = tf.one_hot(self.all_actions, self.output_dims)
                logs = tf.reduce_sum(log_probs * one_hot, axis=1)
                loss = -tf.reduce_mean(logs * reward_tensor)
            grad = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.policy.trainable_variables))
            print('Epoch {}: Reward = {}'.format(epoch, total_reward/episodes_per_epoch))

    def generate_data(self):
        observation = self.env.reset()
        done = False
        rewards = []
        while not done:
            action_logits = self.policy(tf.expand_dims(observation, 0))
            action = tf.random.categorical(tf.nn.log_softmax(action_logits), 1)[0][0].numpy()
            self.all_states.append(observation.copy())
            self.all_actions.append(action)
            observation, reward, done, _ = self.env.step(action)
            rewards.append(reward)
        for i in range(len(rewards)-1, 0, -1):
            rewards[i-1] += rewards[i]
        self.all_rewards += rewards
        return rewards[0]

vpg = VPG("CartPole-v0", [50])
vpg.train(100000, 10)
