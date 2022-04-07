#!/usr/bin/env python

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
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten

from static_board_evaluation import *
from model_trainer import pick_first_best_move, mcts

def get_human_move(board):
	while 1:
		try:
			move = chess.Move.from_uci(input("your move :"))
			if not move in board.legal_moves: raise Exception("Illegal move")
			break
		except:
			print("Incorrect move. Must be legal and UCI format")
	return move

model_side = chess.BLACK
model = load_model("2022-03-18_16-22-38-205912_1200.h5")
board = chess.Board()
max_depth=10
max_pos_nbr=100000
max_search_time=20
depth_penalty=0.3
gamma=0.9
random_factor=0
one_hot=True
verbose=True

if model_side == chess.BLACK:
	print(board)

while not board.is_game_over(claim_draw=True):
	if model_side == chess.WHITE:
		#move, eval = pick_first_best_move(model, board, 0)
		move, pos_nbr, eval = mcts(model=model, board=board, max_depth=max_depth, max_pos_nbr=max_pos_nbr, max_search_time=max_search_time, depth_penalty=depth_penalty, gamma=gamma, random_factor=random_factor, one_hot=one_hot, verbose=verbose)
		board.push(move)
		print(board)
		print(f"Model move : {move.uci()}, eval : {eval}")
		if board.is_game_over(claim_draw=True):
			break
		move = get_human_move(board)
		board.push(move)
	
	else:
		move = get_human_move(board)
		board.push(move)
		if board.is_game_over(claim_draw=True):
			break
		board.apply_mirror()
		#move, eval = pick_first_best_move(model, board, 0)
		move, pos_nbr, eval = mcts(model=model, board=board, max_depth=max_depth, max_pos_nbr=max_pos_nbr, max_search_time=max_search_time, depth_penalty=depth_penalty, gamma=gamma, random_factor=random_factor, one_hot=one_hot, verbose=verbose)
		board.push(move)
		board.apply_mirror()
		print(board)
		print(f"Model move : {move.uci()}, eval : {eval}")


print(board.outcome(claim_draw=True))