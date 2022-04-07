#!/usr/bin/env python

"""static_board_evaluation.py: Evaluates how good the position for each player"""

__author__ = "Aurélien Bück-Kaeffer"
__license__ = "MIT"
__version__= "0.0.1"

import chess
import numpy as np

#Piece square tables, from https://adamberent.com/2019/03/02/piece-square-table/, https://rustic-chess.org/evaluation/psqt.html, 
#All values in centipawns
#For white pieces

pawn_table_w = np.array([0,      0,      0,      0,      0,      0,      0,      0,
5,      10,     10,     -25,    -25,    10,     10,     5,
5,      -5,     -10,    0,      0,      -10,    -5,     5,
0,      0,      0,      25,     25,     0,      0,      0,
5,      5,      10,     27,     27,     10,     5,      5,
10,     10,     20,     30,     30,     20,     10,     10,
50,     50,     50,     50,     50,     50,     50,     50,
0,      0,      0,      0,      0,      0,      0,      0])

pawn_table_w_late = np.array([0,      0,      0,      0,      0,      0,      0,      0,
-10,	-10,	-10,	-10,	-10,	-10,	-10,	-10,
0,		0,		0,		0,		0,		0,		0,		0,
20,		20,		20,		20,		20,		20,		20,		20,
40,		40,		40,		40,		40,		40,		40,		40,
75,		75,		75,		75,		75,		75,		75,		75,
115,	115,	115,	115,	115,	115,	115,	115,
0,		0,		0,		0,		0,		0,		0,		0])

knight_table_w = np.array([-50,    -40,    -20,    -30,    -30,    -20,    -40,    -50,
-40,    -20,    0,      5,      5,      0,      -20,    -40,
-30,    5,      10,     15,     15,     10,     5,      -30,
-30,    0,      15,     20,     20,     15,     0,      -30,
-30,    5,      15,     20,     20,     15,     5,      -30,
-30,    0,      10,     15,     15,     10,     0,      -30,
-40,    -20,    0,      0,      0,      0,      -20,    -40,
-50,    -40,    -30,    -30,    -30,    -30,    -40,    -50])

bishop_table_w = np.array([-20,    -10,    -40,    -10,    -10,    -40,    -10,    -20,
-10,    5,      0,      0,      0,      0,      5,      -10,
-10,    10,     10,     10,     10,     10,     10,     -10,
-10,    0,      10,     10,     10,     10,     0,      -10,
-10,    5,      5,      10,     10,     5,      5,      -10,
-10,    0,      5,      10,     10,     5,      0,      -10,
-10,    0,      0,      0,      0,      0,      0,      -10,
-20,    -10,    -10,    -10,    -10,    -10,    -10,    -20])

king_early_table_w = np.array([20,     30,     10,     0,      0,      10,     30,     20,
20,     20,     0,      0,      0,      0,      20,     20,
-10,    -20,    -20,    -20,    -20,    -20,    -20,    -10,
-20,    -30,    -30,    -40,    -40,    -30,    -30,    -20,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30])

king_late_table_w = np.array([-50,    -30,    -30,    -30,    -30,    -30,    -30,    -50,
-30,    -30,    0,      0,      0,      0,      -30,    -30,
-30,    -10,    20,     30,     30,     20,     -10,    -30,
-30,    -10,    30,     40,     40,     30,     -10,    -30,
-30,    -10,    30,     40,     40,     30,     -10,    -30,
-30,    -10,    20,     30,     30,     20,     -10,    -30,
-30,    -20,    -10,    0,      0,      -10,    -20,    -30,
-50,    -40,    -30,    -20,    -20,    -30,    -40,    -50])

rook_table_w = np.array([0,      0,      0,      10,     10,     10,     0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
15,     15,     15,     20,     20,     15,     15,     15,
0,      0,      0,      0,      0,      0,      0,      0])

queen_table_w = np.array([-20,    -10,    -10,    0,      0,      -10,    -10,    -20,
-10,    0,      5,      0,      0,      0,      0,      -10,
-10,    5,      5,      5,      5,      5,      0,      -10,
-5,     0,      5,      5,      5,      5,      0,      -5,
-5,     0,      5,      5,      5,      5,      0,      -5,
-10,    0,      5,      5,      5,      5,      0,      -10,
-10,    0,      0,      0,      0,      0,      0,      -10,
-20,    -10,    -10,    -5,     -5,     -10,    -10,    -20])

#Lookup tables for black pieces

pawn_table_b = np.array([0,      0,      0,      0,      0,      0,      0,      0,
50,     50,     50,     50,     50,     50,     50,     50,
10,     10,     20,     30,     30,     20,     10,     10,
5,      5,      10,     27,     27,     10,     5,      5,
0,      0,      0,      25,     25,     0,      0,      0,
5,      -5,     -10,    0,      0,      -10,    -5,     5,
5,      10,     10,     -25,    -25,    10,     10,     5,
0,      0,      0,      0,      0,      0,      0,      0])

pawn_table_b_late = np.array([0,      0,      0,      0,      0,      0,      0,      0,
115,	115,	115,	115,	115,	115,	115,	115,
75,		75,		75,		75,		75,		75,		75,		75,
40,		40,		40,		40,		40,		40,		40,		40,
20,		20,		20,		20,		20,		20,		20,		20,
0,		0,		0,		0,		0,		0,		0,		0,
-10,	-10,	-10,	-10,	-10,	-10,	-10,	-10,
0,      0,      0,      0,      0,      0,      0,      0])

knight_table_b = np.array([-50,    -40,    -30,    -30,    -30,    -30,    -40,    -50,
-40,    -20,    0,      0,      0,      0,      -20,    -40,
-30,    0,      10,     15,     15,     10,     0,      -30,
-30,    5,      15,     20,     20,     15,     5,      -30,
-30,    0,      15,     20,     20,     15,     0,      -30,
-30,    5,      10,     15,     15,     10,     5,      -30,
-40,    -20,    0,      5,      5,      0,      -20,    -40,
-50,    -40,    -20,    -30,    -30,    -20,    -40,    -50])

bishop_table_b = np.array([-20,    -10,    -10,    -10,    -10,    -10,    -10,    -20,
-10,    0,      0,      0,      0,      0,      0,      -10,
-10,    0,      5,      10,     10,     5,      0,      -10,
-10,    5,      5,      10,     10,     5,      5,      -10,
-10,    0,      10,     10,     10,     10,     0,      -10,
-10,    10,     10,     10,     10,     10,     10,     -10,
-10,    5,      0,      0,      0,      0,      5,      -10,
-20,    -10,    -40,    -10,    -10,    -40,    -10,    -20])

king_early_table_b = np.array([-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-30,    -40,    -40,    -50,    -50,    -40,    -40,    -30,
-20,    -30,    -30,    -40,    -40,    -30,    -30,    -20,
-10,    -20,    -20,    -20,    -20,    -20,    -20,    -10,
20,     20,     0,      0,      0,      0,      20,     20,
20,     30,     10,     0,      0,      10,     30,     20])

king_late_table_b = np.array([-50,    -40,    -30,    -20,    -20,    -30,    -40,    -50,
-30,    -20,    -10,    0,      0,      -10,    -20,    -30,
-30,    -10,    20,     30,     30,     20,     -10,    -30,
-30,    -10,    30,     40,     40,     30,     -10,    -30,
-30,    -10,    30,     40,     40,     30,     -10,    -30,
-30,    -10,    20,     30,     30,     20,     -10,    -30,
-30,    -30,    0,      0,      0,      0,      -30,    -30,
-50,    -30,    -30,    -30,    -30,    -30,    -30,    -50])

rook_table_b = np.array([0,      0,      0,      0,      0,      0,      0,      0,
15,     15,     15,     20,     20,     15,     15,     15,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      0,      0,      0,      0,      0,
0,      0,      0,      10,     10,     10,     0,      0])

queen_table_b = np.array([-20,    -10,    -10,    -5,     -5,     -10,    -10,    -20,
-10,    0,      0,      0,      0,      0,      0,      -10,
-10,    0,      5,      5,      5,      5,      0,      -10,
-5,     0,      5,      5,      5,      5,      0,      -5,
-5,     0,      5,      5,      5,      5,      0,      -5,
-10,    5,      5,      5,      5,      5,      0,      -10,
-10,    0,      5,      0,      0,      0,      0,      -10,
-20,    -10,    -10,    0,      0,      -10,    -10,    -20])

piece_values = {"Pawn" : 100, "Knight" : 320, "Bishop" : 330, "Rook" : 500, "Queen" : 900, "King" : 20000}#Values in centipawns

"""board = chess.Board()
board.push(chess.Move.from_uci("a2a3"))
board.push(chess.Move.from_uci("b8c6"))
board.push(chess.Move.from_uci("g1f3"))
board.push(chess.Move.from_uci("c6b4"))
board.push(chess.Move.from_uci("f3g1"))
board.push(chess.Move.from_uci("b4d3"))
print([x for x in board.pieces(chess.KNIGHT, chess.BLACK)])"""

def naive_evaluate_position(board):
	"""Gives an estimation of the advantage of each player given a board.
	Positive means white has the advantage, while negative means black has the advantage.
	This does NOT dive into continuations"""

	score = 0
	outcome = board.outcome(claim_draw = False)
	if outcome != None :
		if outcome.winner == chess.WHITE: return piece_values["King"]
		elif outcome.winner == chess.BLACK: return -piece_values["King"]
		else: return 0

	pawn_w = list(board.pieces(chess.PAWN, chess.WHITE))
	knight_w = list(board.pieces(chess.KNIGHT, chess.WHITE))
	bishop_w = list(board.pieces(chess.BISHOP, chess.WHITE))
	rook_w = list(board.pieces(chess.ROOK, chess.WHITE))
	queen_w = list(board.pieces(chess.QUEEN, chess.WHITE))

	pawn_b = list(board.pieces(chess.PAWN, chess.BLACK))
	knight_b = list(board.pieces(chess.KNIGHT, chess.BLACK))
	bishop_b = list(board.pieces(chess.BISHOP, chess.BLACK))
	rook_b = list(board.pieces(chess.ROOK, chess.BLACK))
	queen_b = list(board.pieces(chess.QUEEN, chess.BLACK))

	score += piece_values["Knight"]*len(knight_w) + knight_table_w[knight_w].sum()
	score += piece_values["Bishop"]*len(bishop_w) + bishop_table_w[bishop_w].sum()
	score += piece_values["Rook"]*len(rook_w) + rook_table_w[rook_w].sum()
	score += piece_values["Queen"]*len(queen_w) + queen_table_w[queen_w].sum()

	score -= piece_values["Knight"]*len(knight_b) + knight_table_b[knight_b].sum()
	score -= piece_values["Bishop"]*len(bishop_b) + bishop_table_b[bishop_b].sum()
	score -= piece_values["Rook"]*len(rook_b) + rook_table_b[rook_b].sum()
	score -= piece_values["Queen"]*len(queen_b) + queen_table_b[queen_b].sum()
	
	"""score += sum([piece_values["Pawn"] + pawn_table_w[x] for x in pawn_w])
	score += sum([piece_values["Knight"] + knight_table_w[x] for x in knight_w])
	score += sum([piece_values["Bishop"] + bishop_table_w[x] for x in bishop_w])
	score += sum([piece_values["Rook"] + rook_table_w[x] for x in rook_w])
	score += sum([piece_values["Queen"] + queen_table_w[x] for x in queen_w])
	
	score -= sum([piece_values["Pawn"] + pawn_table_b[x] for x in pawn_b])
	score -= sum([piece_values["Knight"] + knight_table_b[x] for x in knight_b])
	score -= sum([piece_values["Bishop"] + bishop_table_b[x] for x in bishop_b])
	score -= sum([piece_values["Rook"] + rook_table_b[x] for x in rook_b])
	score -= sum([piece_values["Queen"] + queen_table_b[x] for x in queen_b])"""
	
	total_pieces = len(pawn_w) + len(knight_w) + len(bishop_w) + len(rook_w) + len(queen_w) + len(pawn_b) + len(knight_b) + len(bishop_b) + len(rook_b) + len(queen_b)

	if total_pieces > 12:	#We use a different table for the king depending on if we are in the early or endgame
		score += king_early_table_w[list(board.pieces(chess.KING, chess.WHITE))].sum()
		score +- king_early_table_b[list(board.pieces(chess.KING, chess.BLACK))].sum()

		
		score += piece_values["Pawn"]*len(pawn_w) + pawn_table_w[pawn_w].sum()
		score -= piece_values["Pawn"]*len(pawn_b) + pawn_table_b[pawn_b].sum()
		"""score += sum([king_early_table_w[x] for x in board.pieces(chess.KING, chess.WHITE)])
		score -= sum([king_early_table_b[x] for x in board.pieces(chess.KING, chess.BLACK)])"""
	else:
		score += king_late_table_w[list(board.pieces(chess.KING, chess.WHITE))].sum()
		score +- king_late_table_b[list(board.pieces(chess.KING, chess.BLACK))].sum()
		
		
		score += piece_values["Pawn"]*len(pawn_w) + pawn_table_w_late[pawn_w].sum()
		score -= piece_values["Pawn"]*len(pawn_b) + pawn_table_b_late[pawn_b].sum()
		"""score += sum([king_late_table_w[x] for x in board.pieces(chess.KING, chess.WHITE)])
		score -= sum([king_late_table_b[x] for x in board.pieces(chess.KING, chess.BLACK)])"""

	#score += sum([50 if board.piece_at(x).color == True else -50 for x in board.checkers()]) #Counts a piece giving a check as worth 50 centipawns

	return score

#print(naive_evaluate_position(board))

"""for y in range(8):
	s = ""
	for x in range(8):
		s += f"{queen_table_w[(7-y)*8+x]},\t"
	print(s[:-1])"""