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
from random import random, randint
from math import log
from itertools import count

from keras import layers
from keras.models import Sequential, clone_model
from keras.layers.core import Dense, Dropout, Activation, Flatten

from static_board_evaluation import *

tiebreaker = count()

def make_model(state_shape):
	model = Sequential(
		[
			layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", input_shape = state_shape, activation='relu'),
			layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			#layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="valid", activation='relu'),
			#layers.Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding="valid", activation='relu'),
			#layers.Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding="valid", activation='relu'),
			#layers.Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding="valid", activation='relu'),
			layers.Flatten(),
			layers.Dense(8*8, activation = "linear"),
			layers.Dense(1, activation = "linear")
		]
	)
	"""model = Sequential(
    [
        layers.Dense(6*64, activation="linear", input_shape = state_shape),
		layers.Flatten(),
        layers.Dense(64, activation="linear"),
		layers.Dense(64, activation="linear"),
        layers.Dense(64, activation="linear"),
		layers.Dense(64, activation="linear"),
        layers.Dense(1, activation="linear"),
    ])"""
	model.compile(loss='mse', optimizer='Adam',  metrics = ['mae'])
	return model

def save_model(model, epoch):
	model.save("agents/{}".format(datetime.datetime.now()).replace(".", "-").replace(":", "-") + " " + str(epoch) + ".h5")
	print("saved agent")

def save_training_data(filename, epoch, loss, val_loss, mae, val_mae):
	with open(filename, "a") as f:
		f.write(f"{epoch},{loss},{val_loss},{mae},{val_mae}\n")

def myprint(s, filename):
    with open(filename,'a') as f:
        print(s, file=f)

def board_to_np_array(board, one_hot):
	"""Takes a chess.board as input and returns a numpy array"""
	#Pawn = 1
	#Knight = 2
	#Bishop = 3
	#Rook = 4
	#Queen = 5
	#King = 6
	#White = positive, Black = Negative
	#Might wanna consider one hot encoding instead but that would make a massive number of inputs

	if one_hot:
		result = np.zeros((64, 6))

		for x in board.pieces(chess.PAWN, chess.WHITE): result[x][0] = 1
		for x in board.pieces(chess.KNIGHT, chess.WHITE): result[x][1] = 1
		for x in board.pieces(chess.BISHOP, chess.WHITE): result[x][2] = 1
		for x in board.pieces(chess.ROOK, chess.WHITE): result[x][3] = 1
		for x in board.pieces(chess.QUEEN, chess.WHITE): result[x][4] = 1
		for x in board.pieces(chess.KING, chess.WHITE): result[x][5] = 1

		for x in board.pieces(chess.PAWN, chess.BLACK): result[x][0] = -1
		for x in board.pieces(chess.KNIGHT, chess.BLACK): result[x][1] = -1
		for x in board.pieces(chess.BISHOP, chess.BLACK): result[x][2] = -1
		for x in board.pieces(chess.ROOK, chess.BLACK): result[x][3] = -1
		for x in board.pieces(chess.QUEEN, chess.BLACK): result[x][4] = -1
		for x in board.pieces(chess.KING, chess.BLACK): result[x][5] = -1
	
	else:
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
	
	return np.reshape(result, (8, 8, 6))


class node():
	def __init__(self, board, treesearch_prio, depth, parent, value = 0):
		self.board = board
		self.treesearch_prio = treesearch_prio	#priority to give to that node in the search without depth penalty. With depth penalty is calculated by treesearch_prio*(depth_penalty**(depth-1))
		self.depth = depth
		self.parent = parent
		self.children = []
		self.value = value
		self.children_number = 0
	
	def calculate_value(self, gamma):
		naive_val = naive_evaluate_position(self.board)
		if len(self.children)==0:
			self.value = naive_val
		elif self.board.turn == chess.WHITE:
			self.value = naive_val + gamma*max([x.calculate_value(gamma) for x in self.children])
		else:
			self.value = naive_val + gamma*min([x.calculate_value(gamma) for x in self.children])
		return self.value
	
	def count_children(self):
		self.children_number = sum([1+x.count_children() for x in self.children])
		return self.children_number


def append_moves(model, nd, pos_nbr, priority_queue, depth_penalty, random_factor, max_depth, one_hot):
	if max_depth != None and nd.depth+1 > max_depth:
		return pos_nbr, priority_queue
	global tiebreaker
	listNextStates = []
	for x in nd.board.legal_moves:
		nd.board.push(x)
		pos_nbr += 1
		if pos_nbr%250 == 0:
			print(f"pos_nbr : {pos_nbr}", end="\r")
		listNextStates.append(board_to_np_array(nd.board.copy(), one_hot))
		if nd.board.is_checkmate():
			nd.children.append(node(nd.board.copy(), 2**16, nd.depth+1, nd))
			nd.board.pop()
			return pos_nbr, priority_queue
		nd.board.pop()
	
	predictions = model(np.array(listNextStates)).numpy()

	for index, move in enumerate(nd.board.legal_moves):
		#Priority in the treesearch
		treesearch_prio = predictions[index]*(depth_penalty**nd.depth)
		treesearch_prio += treesearch_prio*(0.5-random())*random_factor*2
		if nd.board.turn == chess.WHITE: treesearch_prio *= -1

		#Add node to the heap and tree
		tmp = nd.board.copy()
		tmp.push(move)
		new_node = node(tmp, predictions[index], nd.depth+1, nd)
		heapq.heappush(priority_queue, (treesearch_prio, next(tiebreaker), new_node))
		nd.children.append(new_node)
	
	return pos_nbr, priority_queue

def mcts(model, board, max_depth=None, max_pos_nbr=1000, max_search_time=None, depth_penalty=0.95, gamma=0.9, random_factor=0.2, one_hot = True):
	#TODO: Monte carlos tree search
	tree_root = node(board, 0, 0, None)
	priority_queue = []#Each element in the priority queue is of the form (priority, tiebreaker, node)
	#considered_pos = {}#Hashmap of the already considered pos
	pos_nbr = 0
	
	pos_nbr, priority_queue = append_moves(model, tree_root, pos_nbr, priority_queue, depth_penalty, random_factor, max_depth, one_hot)
	
	while pos_nbr < max_pos_nbr and len(priority_queue) != 0:
		b = heapq.heappop(priority_queue)
		pos_nbr, priority_queue = append_moves(model, b[2], pos_nbr, priority_queue, depth_penalty, random_factor, max_depth, one_hot)
	
	print("Calculating value...", end="\r")
	tree_root.calculate_value(gamma)
	tree_root.children.sort(key=lambda x: x.value)

	print("Counting children...", end = "\r")
	tree_root.count_children()

	#for x in tree_root.children:
	#	print(x.board.peek(), x.value, x.children_number)
	return tree_root.children[-1].board.peek(), pos_nbr, tree_root.children[-1].value

def pick_first_best_move(model, board, random_factor, one_hot):
	listNextStates = []
	listmoves = []
	for x in board.legal_moves:
		board.push(x)
		listNextStates.append(board_to_np_array(board.copy(), one_hot))
		if board.is_checkmate():
			board.pop()
			return x, piece_values["King"]
		board.pop()
		listmoves.append(x)
	
	predictions = model(np.array(listNextStates)).numpy()
	for x in range(len(predictions)):
		predictions[x] += predictions[x]*(0.5-random())*random_factor*2
	amax = np.argmax(predictions)
	return listmoves[amax], predictions[amax]

def play(model, max_depth=None, max_pos_nbr=1000, max_search_time=None, depth_penalty=0.95, gamma=0.9, random_factor=0.2, one_hot = True):
	board = chess.Board()
	boards = []
	positions_w = []
	positions_b = []
	mirrored = False
	while not board.is_game_over(claim_draw=True):
		#best_move, pos_nbr = mcts(model, board, max_depth, max_pos_nbr, max_search_time, depth_penalty, gamma, random_factor)
		best_move, eval = pick_first_best_move(model, board, random_factor, one_hot)
		#print(best_move)
		board.push(best_move)

		if mirrored:
			positions_b.append(board_to_np_array(board, one_hot))
		else:
			positions_w.append(board_to_np_array(board, one_hot))
			boards = [board.fen()] + boards

		board.apply_mirror()

		if mirrored:
			positions_w.append(board_to_np_array(board, one_hot))
			boards = [board.fen()] + boards
		else:
			positions_b.append(board_to_np_array(board, one_hot))

		mirrored = not mirrored

	if mirrored:
		board.apply_mirror()
	
	return board, positions_w, positions_b, boards

def play_mcts(model1, model2, one_hot):
	board = chess.Board()
	move_nbr = 0
	while not board.is_game_over(claim_draw=True):
		move, pos_nbr, eval = mcts(model1, board, 10, 200, None, 0.95, 0.9, 0, one_hot)
		board.push(move)
		move_nbr += 1
		if board.is_game_over(claim_draw=True):
			break
		board.apply_mirror()
		move, pos_nbr, eval = mcts(model2, board, 10, 200, None, 0.95, 0.9, 0, one_hot)
		board.push(move)
		move_nbr += 1
		board.apply_mirror
	return board, move_nbr

def train(model, epochs, batch_size, gamma=0.9):
	#TODO: Main loop. Trains the model by making it play against itself and updating the weights
	#The Q-value of each position n is Q(n) = r + lambda*Q(n+1)
	#With r the naive board evaluation of the current position and lambda the discouted future
	print("Starting training")

	training_data_filename = f"training_data/{datetime.datetime.now()}".replace(".", "-").replace(":", "-") + " " + f"batchsize {batch_size} gamma {gamma}" + ".csv"
	model.summary(print_fn=lambda x : myprint(x, training_data_filename))
	save_training_data(training_data_filename, "epoch", "loss", "val_loss", "mae", "val_mae")

	current_epoch = 0
	previous_save = 0
	total_data = []
	total_labels = []
	#generation = 0
	#best_model = clone_model(model)
	while current_epoch <= epochs:

		if current_epoch - previous_save >= 100:
			previous_save = current_epoch
			save_model(model, current_epoch)

		new_data = []
		labels = []
		for x in range(batch_size):
			current_epoch += 1
			rf = 2 if current_epoch == 1 else 1/log(current_epoch)#Function looks nice I guess, kinda picked it at random
			game, positions_w, positions_b, boards = play(model, max_depth=None, max_pos_nbr=1, max_search_time=None, depth_penalty=0.95, gamma=gamma, random_factor=rf, one_hot=True)

			print(f"Game {current_epoch}: {game.outcome(claim_draw = True)} ({len(positions_w)} moves)")

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
		
		old_data = []
		old_labels = []
		for x in range(int(len(total_data)/10)):
			index = int(random()*len(total_data))
			old_data.append(total_data[index])
			old_labels.append(total_labels[index])
		if len(old_data):
			print(len(old_data))
			training_set = np.concatenate((new_data, old_data), axis = 0)
			training_labels = np.concatenate((labels, old_labels), axis = 0)
		else:
			training_set = np.array(new_data)
			training_labels = np.array(labels)
		
		print(len(training_set), len(training_labels))
		history = model.fit(training_set, training_labels, epochs=5, verbose=1, validation_split=0.1)
		save_training_data(training_data_filename, current_epoch, history.history['loss'][-1], history.history["val_loss"][-1], history.history["mae"][-1], history.history["val_mae"][-1])

		"""if current_epoch/batch_size % 5 == 0:
			models_battle, move_nbr = play_mcts(model, best_model)
			print(f"Result of gen fight : {models_battle.outcome()} ({move_nbr} moves)")
			if models_battle.outcome(claim_draw=True).winner == chess.WHITE:
				generation += 1
				best_model = clone_model(model)
			else:
				print("No progress. Backtracking")
				model = clone_model(best_model)
				model.compile(loss='mse', optimizer='Adam',  metrics = ['mae'])
				current_epoch -= 5*batch_size
			print(f"Current generation : {generation}")"""
		total_data = total_data + new_data
		total_labels = total_labels + labels

		"""to_del = randint(0, 9)
		for x in range(len(total_data)-to_del, -1, to_del):
			del total_data[x]
			del total_labels[x]"""

if __name__ == "__main__":
	model = make_model((8, 8, 6))
	model.summary()
	epochs = 10000
	batch_size = 10
	gamma = 0.9

	train(model=model, epochs=epochs, batch_size=batch_size, gamma=gamma)