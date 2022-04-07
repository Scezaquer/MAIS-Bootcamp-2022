#main.py
from flask import Flask, render_template, request, redirect, url_for
from model_trainer import *
from keras.models import load_model

app = Flask(__name__)
line_mapping_letter_to_num = {"one" : "1", "two" : "2", "three" : "3", "four" : "4", "five" : "5", "six" : "6", "seven" : "7", "eight" : "8"}
line_mapping_num_to_letter = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]

model_side = chess.BLACK
max_depth=10
max_pos_nbr=100000
max_search_time=1
depth_penalty=0.3
gamma=0.9
random_factor=0
one_hot=True
verbose=False

def getwinner(board):
  outcome = board.outcome(claim_draw = True)

  #0 = draw, 1 = white, 2 = black
  winner = "0" if outcome.termination == chess.Termination.STALEMATE else "1"
  winner = "2" if outcome.winner == False else winner

  return winner

@app.route('/')
def home():
  return render_template("landing_page.html")

@app.route('/move', methods=['POST'])
def moved():
  #Get the move played
  fen = request.form.get("board")
  origin = request.form.get("ori")[6:].split(" ")
  destination = request.form.get("dest")[6:].split(" ")
  promotion = request.form.get("prom")
  print(fen, origin, destination, promotion)

  #Load the model
  model = load_model("static/2022-03-18_16-22-38-205912_1200.h5")#takes about half a second to load but is thread safe, not worth optimizing

  #Push the move
  board = chess.Board(fen)

  move = origin[0] + line_mapping_letter_to_num[origin[1]] + destination[0] + line_mapping_letter_to_num[destination[1]] + promotion
  board.push(chess.Move.from_uci(move))

  if board.is_game_over(claim_draw = True): return getwinner(board)

  #Get the model answer
  b = board.mirror()
  move, pos_nbr, eval = mcts(model=model, board=b, max_depth=max_depth, max_pos_nbr=max_pos_nbr, max_search_time=max_search_time, depth_penalty=depth_penalty, gamma=gamma, random_factor=random_factor, one_hot=one_hot, verbose=verbose)
  b.push(move)
  b.apply_mirror()

  for x in board.legal_moves:
    board.push(x)
    if board.board_fen() == b.board_fen():
      move = x
      break
    board.pop()
  board.pop()

  is_en_passant = board.is_en_passant(move)
  is_castle = board.is_castling(move)

  board.push(move)
  print(board)
  print(move.uci())

  #Compose the string to send to the webpage
  #TODO: Move highlighters
  #TODO: Promotion

  #Board
  return_string = board.fen() + "|"

  #Move played
  mv = move.uci()
  return_string += "piece" + " " + mv[0] + " " + line_mapping_num_to_letter[int(mv[1])] + "-"
  return_string += "piece" + " " + mv[2] + " " + line_mapping_num_to_letter[int(mv[3])]
  return_string += "" if len(mv) == 4 else ("-" + (mv[4].upper() if board.turn == False else mv[4]) )#Promotion
  return_string += "-e" if is_en_passant else ""        #En passant
  return_string += "-c" if is_castle else ""            #Caltle
  return_string += "|"

  #Legal moves
  for x in board.legal_moves:
    mv = x.uci()

    if mv[-1] != "q" and len(mv)>4:
      continue

    return_string += "piece" + " " + mv[0] + " " + line_mapping_num_to_letter[int(mv[1])] + "-"
    return_string += "piece" + " " + mv[2] + " " + line_mapping_num_to_letter[int(mv[3])]
    if mv[-1] == "q":
      return_string += "-p"
    return_string += "#"
  
  return_string = return_string[:-1] #remove the last "#" not to have an empty element

  if board.is_game_over(claim_draw=True): return_string += "|" + getwinner(board)

  return return_string

if __name__ == '__main__':
  app.run(debug=True)