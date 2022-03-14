#!/usr/bin/env python

"""
model_trainer.py:
This is the code that generates new models.
Each model predicts the win probability in each position from
white's perspective (we can mirror the board to have it play as black).
Each position is then placed into a priority-queue. Its priority is determined
by how likely the model thinks the position is to lead to a win, and
inversely proportional to how deep this position is in the tree-search.
So the most promising, nearest positions are considered first.
We do not add a position in the queue if it already is in it or if it has already been considered.
(We save already considered or to-consider positions in a hashset for O(1) lookup)

We can decide of a max depth, max number of considered positions, or max search-time.
The best move is then played using a min-max algorithm based on the winning-probabilities
determined by the model.

depth_penalty is a key variable in the algorithm. A smaller depth_penalty will favorize going 
deep into promising variations by considering less moves at each step, while a bigger depth_penalty
will favorize considering more variations at each depth, remaining more at a surface level but
being less likely to miss low depth strategies.

We will use a convolutional neural network, as AlphaZero. This is a choice that was
suited to Go specifically, as the rules are translationally invariant, which is not
the case in chess. However, this seemed to work for them. This does suggest that a
different architecture could be better suited for this task.

The model takes 64 inputs, one for every square. It returns one single output,
that being the probability of white winning in the given position.
"""

__author__ = "Aurélien Bück-Kaeffer"
__license__ = "MIT"
__version__= "0.0.1"

import chess
import tensorflow as tf
import numpy as np
import datetime
import heapq
from random import random
from math import log
from itertools import count

from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from static_board_evaluation import *

tiebreaker = count()

def make_model(state_shape):
	model = Sequential(
    [
        layers.Dense(64, activation="linear", input_shape = state_shape),
        layers.Dense(64, activation="linear"),
		layers.Dense(64, activation="linear"),
        layers.Dense(64, activation="linear"),
		layers.Dense(64, activation="linear"),
        layers.Dense(1, activation="linear"),
    ])
	model.compile(loss='mse', optimizer='Adam',  metrics = ['mae'])
	return model

def save_model(model, epoch):
	model.save("agents/{}".format(datetime.datetime.now()).replace(".", "-").replace(":", "-") + " " + str(epoch) + ".h5")
	print("saved agent")

def board_to_np_array(board):
	"""Takes a chess.board as input and returns a numpy array"""
	#Pawn = 1
	#Knight = 2
	#Bishop = 3
	#Rook = 4
	#Queen = 5
	#King = 6
	#White = positive, Black = Negative
	#Might wanna consider one hot encoding instead but that would make a massive number of inputs

	result = np.zeros(64)

	for x in board.pieces(chess.PAWN, chess.WHITE): result[x] = 1
	for x in board.pieces(chess.KNIGHT, chess.WHITE): result[x] = 2
	for x in board.pieces(chess.BISHOP, chess.WHITE): result[x] = 3
	for x in board.pieces(chess.ROOK, chess.WHITE): result[x] = 4
	for x in board.pieces(chess.QUEEN, chess.WHITE): result[x] = 5
	for x in board.pieces(chess.KING, chess.WHITE): result[x] = 6

	for x in board.pieces(chess.PAWN, chess.BLACK): result[x] = -1
	for x in board.pieces(chess.KNIGHT, chess.BLACK): result[x] = -2
	for x in board.pieces(chess.BISHOP, chess.BLACK): result[x] = -3
	for x in board.pieces(chess.ROOK, chess.BLACK): result[x] = -4
	for x in board.pieces(chess.QUEEN, chess.BLACK): result[x] = -5
	for x in board.pieces(chess.KING, chess.BLACK): result[x] = -6

	return np.array(result)

class node():
	def __init__(self, board, treesearch_prio, depth, parent, value = 0):
		self.board = board
		self.treesearch_prio = treesearch_prio	#priority to give to that node in the search without depth penalty. With depth penalty is calculated by treesearch_prio*(depth_penalty**(depth-1))
		self.depth = depth
		self.parent = parent
		self.children = []
		self.value = value
	
	def calculate_value(self, gamma):
		naive_val = naive_evaluate_position(self.board)
		if len(self.children)==0:
			self.value = naive_val
		elif self.board.turn == chess.WHITE:
			self.value = naive_val + gamma*max([x.calculate_value(gamma) for x in self.children])
		else:
			self.value = naive_val + gamma*min([x.calculate_value(gamma) for x in self.children])
		return self.value


def append_moves(model, nd, pos_nbr, priority_queue, depth_penalty, random_factor):
	global tiebreaker
	for x in nd.board.legal_moves:
		nd.board.push(x)

		"""if nd.board.fen() in considered_pos.keys():#If the board position has already been considered
			nd.board.pop()
			continue"""

		pos_nbr += 1
		if pos_nbr%25 == 0:
			print(f"pos_nbr : {pos_nbr}", end="\r")

		prediction = model.predict(board_to_np_array(nd.board).reshape((1, 64)))
		treesearch_prio = prediction*(depth_penalty**nd.depth)
		treesearch_prio += treesearch_prio*(0.5-random())*random_factor*2
		if nd.board.turn == chess.WHITE: treesearch_prio *= -1

		new_node = node(nd.board.copy(), prediction, nd.depth+1, nd)
		#print(treesearch_prio, tiebreaker)
		heapq.heappush(priority_queue, (treesearch_prio, next(tiebreaker), new_node))
		nd.children.append(new_node)
		#considered_pos[nd.board.fen()] = new_node

		nd.board.pop()
	return pos_nbr, priority_queue

def mcts(model, board, max_depth=None, max_pos_nbr=1000, max_search_time=None, depth_penalty=0.95, gamma=0.9, random_factor=0.2):
	#TODO: Monte carlos tree search
	tree_root = node(board, 0, model.predict(board_to_np_array(board).reshape((1, 64))), None)
	priority_queue = []#Each element in the priority queue is of the form (priority, tiebreaker, node)
	#considered_pos = {}#Hashmap of the already considered pos
	pos_nbr = 0
	
	pos_nbr, priority_queue = append_moves(model, tree_root, pos_nbr, priority_queue, depth_penalty, random_factor)
	
	while pos_nbr < max_pos_nbr:
		b = heapq.heappop(priority_queue)
		pos_nbr, priority_queue = append_moves(model, b[2], pos_nbr, priority_queue, depth_penalty, random_factor)
	
	tree_root.calculate_value(gamma)
	tree_root.children.sort(key=lambda x: x.value)

	return tree_root.children[-1].board.peek(), pos_nbr

def pick_first_best_move(model, board, random_factor):
	listNextStates = []
	listmoves = []
	for x in board.legal_moves:
		board.push(x)
		listNextStates.append(board_to_np_array(board.copy()))
		board.pop()
		listmoves.append(x)
	
	predictions = model.predict(np.array(listNextStates))
	for x in range(len(predictions)):
		predictions[x] += predictions[x]*(0.5-random())*random_factor*2
	return listmoves[np.argmax(predictions)]

def play(model, max_depth=None, max_pos_nbr=1000, max_search_time=None, depth_penalty=0.95, gamma=0.9, random_factor=0.2):
	board = chess.Board()
	boards = []
	positions_w = []
	positions_b = []
	mirrored = False
	while not board.is_game_over(claim_draw=True):
		#best_move, pos_nbr = mcts(model, board, max_depth, max_pos_nbr, max_search_time, depth_penalty, gamma, random_factor)
		best_move = pick_first_best_move(model, board, random_factor)
		#print(best_move)
		board.push(best_move)

		if mirrored:
			positions_b.append(board_to_np_array(board))
		else:
			positions_w.append(board_to_np_array(board))
			boards = [board.fen()] + boards

		board.apply_mirror()

		if mirrored:
			positions_w.append(board_to_np_array(board))
			boards = [board.fen()] + boards
		else:
			positions_b.append(board_to_np_array(board))

		mirrored = not mirrored

	if mirrored:
		board.apply_mirror()
	
	return board, positions_w, positions_b, boards

def train(model, epochs, batch_size, gamma=0.9):
	#TODO: Main loop. Trains the model by making it play against itself and updating the weights
	#The Q-value of each position n is Q(n) = r + lambda*Q(n+1)
	#With r the naive board evaluation of the current position and lambda the discouted future
	print("Starting training")
	current_epoch = 0
	previous_save = -100
	while current_epoch <= epochs:

		if current_epoch - previous_save >= 100:
			previous_save = current_epoch
			save_model(model, current_epoch)

		new_data = []
		labels = []
		for x in range(batch_size):
			current_epoch += 1
			rf = 2 if current_epoch == 1 else 1/log(current_epoch)#Function looks nice I guess, kinda picked it at random
			game, positions_w, positions_b, boards = play(model, max_depth=None, max_pos_nbr=1, max_search_time=None, depth_penalty=0.95, gamma=gamma, random_factor=rf)

			print(f"Game {current_epoch}: {game.outcome()} ({len(positions_w)} moves)")

			boards.pop()

			new_data.append(positions_w[-1])
			labels.append(naive_evaluate_position(game))
			positions_w.pop()

			for y in boards:
				new_data.append(positions_w.pop())
				labels.append(naive_evaluate_position(chess.Board(y)) + gamma*labels[-1])
			
			new_data.append(positions_b[-1])
			labels.append(-1*naive_evaluate_position(game))
			positions_b.pop()

			for y in boards:
				new_data.append(positions_b.pop())
				labels.append(-1*naive_evaluate_position(chess.Board(y)) + gamma*labels[-1])
		
		print(len(np.array(new_data)), len(np.array(labels)))
		model.fit(np.array(new_data), np.array(labels), epochs=5, verbose=1)

model = make_model((64,))
epochs = 10000
batch_size = 10
gamma = 0.9

train(model=model, epochs=epochs, batch_size=batch_size, gamma=gamma)