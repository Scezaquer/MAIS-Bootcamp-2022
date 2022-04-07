# MAIS-Bootcamp-2022
A chess AI using reinforcement learning techniques

# Problem Statement

We are creating a chess engine using reinforcement learning. Our model’s goal is to learn through self-play how to correctly estimate a position’s value based on material advantage, positional advantage, and predicted future positions and outcome. We then use this model to orient a tree search algorithm by prioritizing some game variations over others in our search for good moves, and finally we pick the one that has the best expected result to play.

# Data preprocessing

We are working with games of chess generated through self-play by our model. It was originally suggested to use grandmaster’s games from databases in the training as well, but this has not been done so far. This, however, remains an option, as a similar method was used in AlphaGo [1]. We are using a convolutional neural network, and therefore treating the chess board as an (8×8) image.

We are using one-hot encoding for each case of the chess board: Every case in represented by a vector of dimension 6, with one element for each type of piece, positive being white and negative black. Each vector is of the format [𝑝𝑎𝑤𝑛,𝑘𝑛𝑖𝑔ℎ𝑡,𝑏𝑖𝑠ℎ𝑜𝑝,𝑟𝑜𝑜𝑘,𝑞𝑢𝑒𝑒𝑛,𝑘𝑖𝑛𝑔]. So, a white pawn would be represented as [1,0,0,0,0,0] while a black queen is [0,0,0,0,−1,0]. Therefore, our data’s complete dimensions are (8×8×6).

The number of samples increases as time goes by, as it is generated through self-play. We let the model play 10 games, then generate the Q-value corresponding with each position S using the formula 

𝑄(𝑆)=𝑛𝑎𝑖𝑣𝑒𝐸𝑣𝑎𝑙(𝑆)+𝛾𝑄(𝑆+1)

With 𝑛𝑎𝑖𝑣𝑒𝐸𝑣𝑎𝑙(𝑆) being an evaluation of the value of the current position based on material imbalance, piece positions and if this board is a checkmate/stalemate. It does not consider future positions, hence the name “naïve” (see static_board_evaluation.py for the exact implementation).
𝑄(𝑆+1) is the Q-value of the next position that was played in the game.

𝛾 is a hyperparameter between 0 and 1 that we use to decrease the impact of the value of future positions the further away they are. We set it at 0.9.

# Machine Learning Model

As for Alpha Zero [3], we are using a convolutional neural network as our model for this task. However, this choice was originally made to play Go, since its rules are translationally invariant, which is not the case for chess (castling isn’t the same on both sides, pawns promote once they reach the other side of the board…). This was because the point was to make a general-purpose algorithm and changing the architecture for each problem defeats this purpose. We, however, are looking into chess exclusively, and can therefore tune our architecture to fit it better.

As seen in lectures, convolutional neural networks have pooling layers in order to obtain the translational invariant important in image processing. We, however, do not want this invariant, and therefore removed the pooling layers from the CNN.

The model is implemented using Keras. It is composed of 6 convolutional layers with a kernel of 3×3 and 32 filters as well as padding and relu activation function, one 64 neuron wide dense layer with linear activation, and one output layer containing only one neuron with linear activation as well. Since modifying this architecture yields widely different results, changes in these specifications are to be expected.

Each iteration, training is done on 10% of the previously seen positions to limit catastrophic forgetting [2], on top of all the new positions.

A validation split is done to save 10% of the dataset for assessing the efficiency of the model on new data.

To increase the occurrences of checkmate in the training data for our model to learn to recognize it as the goal, we made playing any checkmate in 1 move that happens in a game mandatory to play. It appears to be sufficient at making non-drawn outcomes appear often.

As stated before, the full algorithm to play a game is a tree search algorithm that uses the model to know in which parts of its search it should focus. We have both a priority queue to determine which node is the next to explore, and a tree. The tree’s root is the current position, and its children are the positions we can reach on the next move, and so on.

Every node is also in the min-heap with a priority value P. Finding a proper equation to calculate P is the key to our algorithm. The lower the P value for a given position, the more our algorithm needs to prioritize visiting it in the tree search. We want P to increase with the depth of the node in our search tree, as moves further away are less worth considering.

Multiplying the network’s prediction by a depth penalty constant 𝛿>1 to the power of the node’s depth looks like a good way of doing this at first, but it runs into an issue: The model predicts real values, both positive and negative. So if it predicts a negative value of -500 for a board at depth 8 for 𝛿=1.05, it will actually attribute it the priority −500∗(1.05)8≈−738.7, which is lower than our original value. But we are in a min heap, which mean we actually increased how important this node is to consider, which is not what we want.

To get rid of this problem, we need a function 𝑓 that turns the negative values positive but keeps the ordering between them as we still want values that used to have a lower priority to keep said lower priority. 𝑓:ℝ→ℝ+ 𝑥>𝑦→𝑓(𝑥)>𝑓(𝑦)

Luckily for us, there exists a function that does exactly that, which is the exponential function. Unfortunately, it has another issue: it gets extremely big extremely quickly, as well as extremely close to 0 extremely fast. Our computer’s precision being limited, this doesn’t work for us. We need to find a way to modify it to fix this property.

The natural logarithm does exactly that for us, but it is only defined on \[0,∞). No worries, we simply need to make a piecewise function out of 2 logarithms: ln(𝑥) and −ln⁡(−𝑥), which covers ℝ∗.
We shift them to get a continuous function
We then define our piecewise continuous function ℎ as ℎ:ℝ→ℝ ℎ(𝑥)={−ln(−𝑥+1)𝑖𝑓⁡𝑥<0ln(𝑥+1)𝑖𝑓⁡𝑥≥0
Now since this function is defined using natural logarithms, we can input it into an exponential function and it will not go uncontrollably towards infinity or 0, but only at a steady pace in both directions, solving our limited precision problem. We can finally find our function 𝑓 that gives the priority of each node to be: 𝑓(𝑥)=𝑒^ℎ(𝑥)

Now this function should be good enough as is, but it still has a few tweaks we can give it. We have a min-heap, so we care more about the small values than about the big values. But we have all our small values cramped together between 0 and 1, while the big values have all the space between 0 and infinity. We can reverse this by having a few well-placed negative signs giving us our final function 𝑓(𝑥)=−𝑒ℎ(−𝑥)

Additionally, it has the advantage of making values we multiply by a number between 0 and 1 get a lesser priority in the min-heap, which is a lot more intuitive and simpler to work with.


Now getting back to our priority value P, the function we use to calculate it is 𝑃=𝑓(𝑦̂∗(−1)𝑤)∗𝛿𝑑

With 𝑦̂ the model’s prediction, 𝑑 the depth of the node in the search tree and 𝛿∈[0,1] the depth penalty that we chose to be 0.3. The deeper a node is, the less priority we give it. This parameter can be tuned: a depth penalty closer to 0 will result in more positions being considered at each step, and therefore a wider tree but that goes into less depth for each variation. On the other hand, a depth penalty closer to 1 will favorize going deep into a few variations, resulting in a narrower, deeper tree, but more likely to miss moves.

The variable 𝑤 is equal to 1 if the side whose turn it is to play is white, and 0 if it is black. This is following a min-max algorithm logic: We want the network to prioritize moves that are good for itself when its his turn to play and consider the best opponent’s moves when its not (meaning, the moves
that are worst for itself). The priority queue is a min-heap so the smaller the priority value, the more we want to consider that move.
We then apply a classical min-max algorithm on the tree we obtained after either a certain number of positions has been considered or a certain amount of time has elapsed and pick the move that had the highest value based on this. We only pick nodes that have at least 1 child (except if it is a checkmate), because we do not want to play moves that our tree search did not dive into.

It is important to note that we do not run the entire algorithm during training, as it is way too slow for it. Instead, we only play the move the model predicts the highest output for without diving any deeper into the tree. Considering games regularly go into the 200 moves before finishing, and with an average of 35 legal moves per position, this is already 7000 positions to run our algorithm on per game. More than this is simply too slow given our processing power.

To train our model, we let it play 10 games against itself, then for every encountered position we mirror the board in order to have both white and black perspective to double the amount of data we get, calculate the Q-value, and fit the model on it plus 10% of the positions seen in previous games for 5 epochs before repeating. To ensure the model plays different games each time, we add a random factor 𝑟𝑓 that decreases over time, and that makes it so that every prediction the model makes gets added a random value within ±𝑟𝑓% of itself, to vary the move chosen and encourage confidence. The move chosen should still be among the top moves, but not always the top one, creating multiple game variations. 𝑟𝑓=1ln(𝑔𝑎𝑚𝑒𝑁𝑢𝑚𝑏𝑒𝑟)

This function decreases very slowly and is between 20% and 10% for most of the training.
However, there is an issue: if the model plays multiple times very similar games during a batch, it will have multiples instances of the same data when training and start to overfit, which will make it play more confidently the same moves and diminish further the likelihood of playing a different move, and the model gets into a self-reinforcing spiral that makes it always play the same game. This happens sometimes during training, but not often.


# Final training results

We have tried different sets of parameters to improve the performances of the model. Mostly, the architecture: the parameter that seems to have the most effect when decreasing the mean average error is the number of filters per convolutional layer.

Drastically upsizing the model from 32 filters per layer to 256, making the model go from 179 201 to 3,095,569 total parameters, has interesting results, as the validation mean absolute error got a very significant improvement, from about 2300 to about 1500 (about 34% improvement). However, this has massive consequences on the length of training and on the number of positions the model is able to consider before making a move.

The biggest model considers approximately 200 positions per second on my machine, which is excruciatingly slow. For reference, the model with 180k parameters can consider about 1450 per second, and a full search algorithm with no neural network can consider about 4700 per second.

To test whether it was better to have a more precise model or a model that can consider a lot of positions, we made both play a match of 12 games, with 6 different book openings (this is important since the algorithm is deterministic. It will always play the same moves if in the same position). Each model played 6 games as white and 6 as black. Each model had about 20s per move to calculate what to play. In total, there were 4 wins by the faster model (2 as black, 2 as white), 2 by the slower (1 as white, 1 as black) and 6 draws, making the faster model winner 7 to 5. We therefore will go with this one for further testing of the results.

A middle point between 180k and 3M parameters could undoubtedly be found, but it is extremely long to do so, especially since training gets slower the more parameters we have, and proper testing takes hours just to compare two models.

It is important to note that the deterministic nature of these algorithms makes them prone to repeating moves, as if a previous position reappears, the exact same sequence of moves that led to it will be repeated. This is a problem that does not exist when playing against non-deterministic bots or humans (though they could exploit this to get an easy draw in a losing position). Since this issue mostly occurs when comparing two models and isn’t significant outside of this scenario, it has been neglected.

Previously, multiple problems have been mentioned:

1- The model tends to play moves that have not been thoroughly considered since they appear better at face value but were not worth diving deeper into as they were too bad.

2- The model tends to play optimally when the position is deemed equal. If a side is clearly winning, it will either consider too many moves per node in the tree or too little and hyperfocus on one. This was due to the inconsistent nature of the priority function in our tree search: linear when a position is good and 1/x when it is not. Combined with the depth penalty, this created the previously described behavior.

Both problems appear to be fixed to some extend by the implementation of normalization in the tree search: We take the average of the model’s predictions for each possible very next move from the root (prediction done without depth, supposed to orient the tree search), and subtract it from every prediction in the tree search.
This way, the algorithm will always behave as if the positions were equal for its best move search, which we observed to be the optimal behavior. This removes the problem of considering too many or too little moves if winning or losing, making the model significantly better at holding disadvantageous positions, and keep its advantage in advantageous ones. This had the involuntary consequence of making a more uniformly distributed tree search among the moves that do get considered by the model, making rare moves that are only considered at depth 1, and consequently effectively removing problem 1.

There still are problems though, mainly difficulties to convert a winning position into a checkmate, which resulted in a few unnecessary draws against low rated opponents. This was improved by the implementation of a special piece-square table for pawns in the naïve evaluation function. By dividing the piece-square table into two parts, one for the early/middle game, and one for the endgame, we can tune the behavior of the algorithm depending on which phase of the game it currently is in. So, by introducing an endgame piece-square table for the pawns that rewards more advanced pawn, we made the model more likely to promote and win a game. This is an interesting path for improvement, since similar things can be introduced for each kind of piece. Note that the model has not been trained using these improved piece-square tables due to the training and testing time implied, so that is also something to possibly investigate.

To test the performance of our model, we made it play against the chess.com bots rated 250 to 2000. Since they are bots, we can assume they play consistently at the same level, and their regular Elo spacing allows for great benchmarking. In total, our model played 32 games against 16 different bots, 2 games against each, one as each color, with no opening book, 20s per move. Full results are in the following table.

![scoretable](https://github.com/Scezaquer/MAIS-Bootcamp-2022/blob/main/scoreboard.png)

We can make a few interesting observations: The model did not lose a single game until the 1100 Elo opponent. However, it did draw a few due to the previously mentioned inability to convert a winning position into checkmate. The turning point between reliably winning and consistently losing appears to be between 1100 and 1200, opponents that both had one win and one loss against our model. The model drew one last game against the 1300 bot before losing every subsequent game against higher rated opponents.

The average opponent rating was about 1203. Using the formula for calculating performance rating, we can get an Elo rating estimation for our final model: 𝑅𝑝=𝑅𝑎+400∗𝑊𝑖𝑛𝑠−𝐿𝑜𝑠𝑠𝑒𝑠𝑛

With 𝑅𝑝 the performance rating, 𝑅𝑎 the average opponent rating, and 𝑛 the total number of games played. 𝑅𝑝=1203+400∗13−1932 𝑅𝑝=1128

If we assume that the bots’ Elo is a good benchmark and that they are properly rated, this puts our model approximately in the 84th percentile of chess.com players. Still, note that chess.com Elo is considerably inflated compared to FIDE rating.

We are exceeding the 1000 performance rating we were hoping for as a baseline objective in deliverable 1. This should hopefully mean that our model can reliably beat new players. However, it is to be expected that humans play vastly differently than bots, making hazardous any prediction on performance against real players.

# References

[1] Silver, D., Huang, A., Maddison, C. et al. Mastering the game of Go with deep neural networks and tree search. Nature 529, 484–489 (2016). https://doi.org/10.1038/nature16961 

[2] Andrew Cahill, Catastrophic Forgetting in Reinforcement-Learning Environments, University of Otago (2010) Catastrophic Forgetting in Reinforcement-Learning Environments (otago.ac.nz) 

[3] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis, A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (2018), Science, 1140-1144, 362, 6419, doi:10.1126/science.aar6404
