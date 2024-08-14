import CIR
import math
import random

MAX_CHILDREN = 3

class Node:
    def __init__(self, question, answer, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) >= MAX_CHILDREN

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda c: (c.value / c.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / c.visits) if c.visits else float('inf'))

    def add_child(self, child_node):
        self.children.append(child_node)

class MCTS:
    def __init__(self, question, seed_answers, iterations=2):
        self.question = question
        self.iterations = iterations
        self.root = Node(question, random.choice(seed_answers))

    def search(self):
        for _ in range(self.iterations):
            node = self.select(self.root)
            if not node.is_fully_expanded():
                node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.best_child(exploration_weight=0).answer

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node):
        child = Node(self.question, node.answer, parent=node)
        node.add_child(child)
        critique = CIR.critique(self.question, child.answer)
        child.answer = CIR.improve_answer(self.question, child.answer, critique)
        return random.choice(node.children)

    def simulate(self, node):
        return CIR.rate_answer(self.question, node.answer)

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent