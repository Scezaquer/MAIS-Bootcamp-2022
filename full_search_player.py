#!/usr/bin/env python

"""full_search_player.py: Engine that considers every single move up to depth n without optimized search"""

__author__ = "Aurélien Bück-Kaeffer"
__license__ = "MIT"
__version__= "0.0.1"

import chess
from static_board_evaluation import *

depth = 3	#Must be at least 1 to return a list of move evaluations, otherwise for depth = 0 it just returns the naive evaluation of the current board

board = chess.Board()

def get_best_move(board, depth):
	if depth == 0 or board.is_game_over():
		return {"current_board" : naive_evaluate_position(board)}

	global moves_considered
	move_eval = {}
	for x in board.legal_moves:
		#Basic minmax algorithm
		moves_considered += 1
		if not moves_considered%1000 : print(f"Moves considered: {moves_considered}", end="\r")
		board.push(x)
		next_move_eval = get_best_move(board, depth-1)
		move_eval[x.uci()] = (min(next_move_eval.values()) if board.turn == True else max(next_move_eval.values()))
		board.pop()
	
	return move_eval

"""moves_considered = 0
move_list = get_best_move(board, depth)
print(f"moves considered: {moves_considered}")

print(move_list)
max_key = max(move_list, key=move_list.get)

print(f"Best move: {max_key} [{move_list[max_key]}]")"""

while not board.is_game_over():
	moves_considered = 0
	move_list = get_best_move(board, depth)
	print(f"moves considered: {moves_considered}")

	print(move_list)
	max_key = max(move_list, key=move_list.get)

	board.push(chess.Move.from_uci(max_key))
	print(board)

	print(f"Best move: {max_key} [{move_list[max_key]}]")

	while 1:
		player_move = input("Your move : ")
		try:
			player_move = chess.Move.from_uci(player_move)
		except:
			print("Move must be valid uci and legal")
			continue
		
		if player_move in board.legal_moves:
			break
		print("Move must be valid uci and legal")
	
	board.push(player_move)