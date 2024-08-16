import CIR  # Importing the CIR module (Critique, Improve, Rate operations)
import math
import random

MAX_CHILDREN = 3  # Maximum number of children allowed for each node

class Node:
    def __init__(self, question, answer, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []  # List to store child nodes
        self.visits = 0  # Number of times this node has been visited
        self.value = 0.0  # Cumulative value of this node

    def is_fully_expanded(self):
        # Check if the node has the maximum number of children
        return len(self.children) >= MAX_CHILDREN

    def best_child(self, exploration_weight=1.41):
        # Select the best child using the UCT (Upper Confidence Bound for Trees) formula
        return max(self.children, key=lambda c: (c.value / c.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / c.visits) if c.visits else float('inf'))

    def add_child(self, child_node):
        # Add a child node to this node's children list
        self.children.append(child_node)

class MCTS:
    def __init__(self, question, seed_answers, iterations=2):
        self.question = question
        self.iterations = iterations
        # Create the root node with a random seed answer
        self.root = Node(question, random.choice(seed_answers))

    def search(self):
        # Main loop of the MCTS algorithm
        for _ in range(self.iterations):
            # Select a node to expand
            node = self.select(self.root)
            # Expand the node if it's not fully expanded
            if not node.is_fully_expanded():
                node = self.expand(node)
            # Simulate the outcome from this node
            reward = self.simulate(node)
            # Propagate the result back up the tree
            self.backpropagate(node, reward)
        # Return the best answer found after all iterations
        return self.root.best_child(exploration_weight=0).answer

    def select(self, node):
        # Select a node to expand using the UCT selection strategy
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node):
        # Create a new child node and add it to the current node
        child = Node(self.question, node.answer, parent=node)
        node.add_child(child)
        # Use CIR module to critique the current answer and improve it
        critique = CIR.critique(self.question, child.answer)
        child.answer = CIR.improve(self.question, child.answer, critique)
        # Return a random child of the expanded node
        return random.choice(node.children)

    def simulate(self, node):
        # Simulate the outcome by rating the answer using the CIR module
        return CIR.rate(self.question, node.answer)

    def backpropagate(self, node, reward):
        # Backpropagate the reward up the tree, updating visit count and value
        while node:
            node.visits += 1  # Increment visit count for each node in the path
            node.value += reward  # Add reward to each node's value
            node = node.parent  # Move up to the parent node