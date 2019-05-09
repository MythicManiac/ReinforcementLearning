import gym
import random

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


env = gym.make("CartPole-v0")


TRUE_FALSE_TUPLE = (True, False)


def crossover_and_mutate(weights_a, weights_b, mutate_chance):
    result = []
    for index in range(len(weights_a)):
        selection = np.random.choice(TRUE_FALSE_TUPLE, weights_a[index].shape)
        crossed_over = np.where(selection, weights_a[index], weights_b[index])

        mutations = np.random.random_sample(weights_a[index].shape)
        entries_to_mutate = np.random.choice(
            TRUE_FALSE_TUPLE,
            weights_a[index].shape,
            [mutate_chance, 1.0 - mutate_chance],
        )
        mutated = np.where(entries_to_mutate, mutations, crossed_over)

        result.append(mutated)
    return result


class Population:

    def __init__(self, population_size, member_cls, mutate_chance):
        self.population_size = population_size
        self.member_cls = member_cls
        self.mutate_chance = mutate_chance
        self.members = []

    def seed_population(self):
        print("Seeding population")
        self.members = []
        for _ in range(self.population_size):
            self.members.append(self.member_cls())

    def evaluate_population(self):
        print("Evaluating population")
        for member in self.members:
            member.evaluate()

    def perform_selection(self):
        print("Performing selection")
        return sorted(self.members, key=lambda x: -x.reward)[:10]

    def crossover_population(self, choices):
        print("Crossing over")
        weight_choices = [x.model.get_weights() for x in choices]
        for index, member in enumerate(self.members):
            member.update_weights(crossover_and_mutate(
                weights_a=random.choice(weight_choices),
                weights_b=random.choice(weight_choices),
                mutate_chance=self.mutate_chance,
            ))


class Model:

    def __init__(self, weights=None):
        self.model = self.build_model()
        if weights:
            self.update_weights(weights)
        self.reward = 0

    def build_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=4))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation("relu"))
        return model

    def update_weights(self, weights):
        self.model.set_weights(weights)

    def evaluate(self):
        global env
        self.reward = 0
        env.reset()
        observation = env.reset()
        for step in range(1000):
            prediction = self.model.predict(observation.reshape(1, 4))
            action = max(min(int(round(prediction[0][0])), 1), 0)
            observation, reward, done, info = env.step(action)
            self.reward += reward
            if done:
                break
        print(f"Finished with {self.reward} reward")

def main():
    population = Population(
        population_size=30,
        member_cls=Model,
        mutate_chance=0.05
    )
    population.seed_population()
    for i in range(1000):
        print("#" * 20)
        print(f"Generation {i}")
        print("#" * 20)
        population.evaluate_population()
        choices = population.perform_selection()
        print(f"Best reward: {choices[0].reward}")
        population.crossover_population(choices)


if __name__ == "__main__":
    main()
