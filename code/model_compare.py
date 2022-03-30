#!/usr/bin/env python

__author__ = "Aurélien Bück-Kaeffer"
__license__ = "MIT"
__version__= "0.0.1"

from model_trainer import *
import chess
from keras.models import load_model

def play_game(board, model_1, model_2, max_search_time, max_pos_nbr, max_depth, depth_penalty, gamma, random_factor, one_hot, verbose, score):
	#Model 1 plays as white
	move_nbr = 0
	while not board.is_game_over(claim_draw=True):
		move, pos_nbr, eval = mcts(model=model_1, board=board, max_depth=max_depth, max_pos_nbr=max_pos_nbr, max_search_time=max_search_time, depth_penalty=depth_penalty, gamma=gamma, random_factor=random_factor, one_hot=one_hot, verbose=verbose)
		board.push(move)
		move_nbr += 1

		with open("Game_results.txt", "a") as f:
			f.write(f"{board}\n{move.uci()}\n\n")
		if verbose: print(f"{board}\nMove: {move.uci()}\nCurrent score : {score}")

		if board.is_game_over(claim_draw=True):
			break
		b = board.mirror()
		move, pos_nbr, eval = mcts(model=model_2, board=b, max_depth=max_depth, max_pos_nbr=max_pos_nbr, max_search_time=max_search_time, depth_penalty=depth_penalty, gamma=gamma, random_factor=random_factor, one_hot=one_hot, verbose=verbose)
		b.push(move)
		b.apply_mirror()

		for x in board.legal_moves:
			board.push(x)
			if board.board_fen() == b.board_fen():
				move = x
				break
			board.pop()

		move_nbr += 1

		with open("Game_results.txt", "a") as f:
			f.write(f"{board}\n{move.uci()}\n\n")
		if verbose: print(f"{board}\nMove: {move.uci()}\nCurrent score : {score}")
	return board


openings = [
	"rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",		#French defense
	"rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",	#Sicilian defense
	"rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",	#Queen's gambit declined
	"rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",	#Slav defense
	"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",#Italian game #3...Nf6
	"rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3" 	#Caro-Kann
]

name_m1 = "2022-03-18_16-22-38-205912_1200.h5"
name_m2 = "2022-03-24 08-00-59-503971 400.h5"
model_1 = load_model(name_m1)
model_2 = load_model(name_m2)

score = {"model_1w": 0, "model_1b": 0, "model_2w": 0, "model_2b": 0, "Draws": 0}

max_depth=10
max_pos_nbr=100000
max_search_time=20
depth_penalty=0.3
gamma=0.9
random_factor=0
one_hot=True
verbose=True

#Model 1 plays as white
for x in openings:
	board = chess.Board(x)
	with open("Game_results.txt", "a") as f:
		f.write(f"\n\nNew game starting with\n{board}\n{name_m1} as white\n{name_m2} as black\n\n")
	board = play_game(board, model_1, model_2, max_search_time, max_pos_nbr, max_depth, depth_penalty, gamma, random_factor, one_hot, verbose, score)
	outcome = board.outcome(claim_draw=True)
	if outcome.winner == None: score["Draws"] += 1
	elif outcome.winner == chess.WHITE: score["model_1w"] += 1
	elif outcome.winner == chess.BLACK: score["model_2b"] += 1
	with open("Game_results.txt", "a") as f:
		f.write(f"\n{outcome}\n")
	print(board.outcome(claim_draw=True))
	print(f"Current score : {score}")


#Model 2 plays as white
for x in openings:
	board = chess.Board(x)
	with open("Game_results.txt", "a") as f:
		f.write(f"\n\nNew game starting with\n{board}\n{name_m2} as white\n{name_m1} as black\n\n")
	board = play_game(board, model_2, model_1, max_search_time, max_pos_nbr, max_depth, depth_penalty, gamma, random_factor, one_hot, verbose, score)
	outcome = board.outcome(claim_draw=True)
	if outcome.winner == None: score["Draws"] += 1
	elif outcome.winner == chess.WHITE: score["model_2w"] += 1
	elif outcome.winner == chess.BLACK: score["model_1b"] += 1
	with open("Game_results.txt", "a") as f:
		f.write(f"\n{outcome}\n")
	print(board.outcome(claim_draw=True))
	print(f"Current score : {score}")
