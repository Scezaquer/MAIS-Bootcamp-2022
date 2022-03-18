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

model_side = chess.WHITE
model = load_model("agents/2022-03-14 15-53-54-791117 2000.h5")
board = chess.Board()
one_hot = True

if model_side == chess.BLACK:
	print(board)

while not board.is_game_over(claim_draw=True):
	if model_side == chess.WHITE:
		#move, eval = pick_first_best_move(model, board, 0)
		move, pos_nbr, eval = mcts(model, board, 10, 10000, None, 0.95, 0.9, 0, one_hot)
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
		move, pos_nbr, eval = mcts(model, board, 10, 1000, None, 0.95, 0.9, 0, one_hot)
		board.push(move)
		board.apply_mirror()
		print(board)
		print(f"Model move : {move.uci()}, eval : {eval}")


print(board.outcome(claim_draw=True))