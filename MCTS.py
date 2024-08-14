import CIR
import math
import random
import numpy as np

max_children = 3

class Node:
    def __init__(self, question, answer, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) >= max_children

    def best_child(self, exploration_weight=1.41):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')  # Prioritize unexplored nodes
            else:
                weight = (child.value / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]

    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visits)

    def add_child(self, child_node):
        self.children.append(child_node)

class MCTS:
    def __init__(self, question, seed_answers, iterations=2):
        self.question = question
        self.seed_answers = seed_answers
        self.iterations = iterations
        self.root = Node(question, random.choice(seed_answers))

    def search(self):
        for i in range(self.iterations):
            print(f"\nIteration {i+1}/{self.iterations}")
            node = self.select(self.root)
            print(f"Selected Node: {node.answer}")
            if not node.is_fully_expanded():
                node = self.expand(node)
                print(f"\nExpanded Node: {node.answer}")
            reward = self.simulate(node)
            print(f"\nSimulated Reward: {reward}")
            self.backpropagate(node, reward)
        print(f"Visits to most visited child: {self.root.most_visited_child().visits}")
        return self.root.most_visited_child().answer

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node):
        for j in range(max_children - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node)  # Start with the same a
            node.add_child(child_node)
            
            critique = CIR.critique(self.question, child_node.answer)
            print(f"\n--Critique {j}--\n{critique}")
            
            improved_answer = CIR.improve_answer(self.question, child_node.answer, critique)
            print(f"\n--Improved answer {j}--\n{improved_answer}")
            
            child_node.answer = improved_answer  # Update the child node's answer with the impr
        return random.choice(node.children)

    def simulate(self, node):
        rating = CIR.rate_answer(self.question, node.answer)
        return rating

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            # print(f"Backpropagating Node: {node.answer}, Visits: {n
            node = node.parent