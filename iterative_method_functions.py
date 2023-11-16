import numpy as np
import random
from scipy.stats import norm
import itertools
import matplotlib.pyplot as plt
from matplotlib import colormaps

def payoff_calculation(player, player_positions, player_count, M, points_per_position_list):
    """
    Caluclate the payoff of a specified player given a strategy combination
    player: player for which we calculate the payoff
    player_positions: dictionary with keys corresponding to players, 
    and value corresponding to their strategy
    player_count: dictionary with keys corresponding to strategies, 
    and values being number of players with this strategy
    normal: boolean value to determine whether payoff is distributed uniformly
    or normally
    score: value of the payoff for the given player
    """

    # giving player points from the position they are standing at
    score = points_per_position_list[player_positions[player]]/player_count[player_positions[player]]

    # looping thorugh the posiitions either side of the player to determine where nearest neighbours are
    left_neighbour, right_neighbour = player_positions[player] - 1, player_positions[player] + 1

    while left_neighbour > -1:
        if player_count[left_neighbour] > 0:
            break
        else:
            left_neighbour -= 1

    while right_neighbour < M:
        if player_count[right_neighbour] > 0:
            break
        else:
            right_neighbour += 1
    
    # calculating score according to were neighbour is on each side
    if left_neighbour == -1:
        score += sum(points_per_position_list[:player_positions[player]]) / player_count[player_positions[player]]
    else:
        if (player_positions[player] - left_neighbour - 1) % 2 == 0:
            split = player_positions[player] - int((player_positions[player] - left_neighbour - 1) / 2)
            score += sum(points_per_position_list[split:player_positions[player]]) / player_count[player_positions[player]]
        else:
            split = player_positions[player] - int((player_positions[player] - left_neighbour - 2) / 2)
            score += sum(points_per_position_list[split:player_positions[player]]) / player_count[player_positions[player]]
            score += points_per_position_list[split - 1] / (2 * player_count[player_positions[player]])
    
    if right_neighbour == M:
        score += sum(points_per_position_list[player_positions[player]+1:]) / player_count[player_positions[player]]
    else:
        if (right_neighbour - player_positions[player] - 1) % 2 == 0:
            split = player_positions[player] + int((right_neighbour - player_positions[player] - 1) / 2)
            score += sum(points_per_position_list[player_positions[player]+1:split+1]) / player_count[player_positions[player]]
        else:
            split = player_positions[player] + int((right_neighbour - player_positions[player] - 2) / 2)
            score += sum(points_per_position_list[player_positions[player]+1:split+1]) / player_count[player_positions[player]]
            score += points_per_position_list[split + 1] / (2 * player_count[player_positions[player]])
    
    return round(score,5)


def election_equilibrium(N = 2, M = 10, nsim = 1000, points_per_position = 10, normal = False, random_sequence = False):
    """
    Find equilibria of the election game by iteratively moving each 
    player to their best response.
    N: number of players (players are labelled 0, 1, ..., N-1)
    M: number of strategies (strategies are labelled 0, 1, ..., M-1)
    nsim: number of simulations to carry out
    equilibria: list of dictionaries, each dictionary specifies positions for an equilibrium
    """

    equilibria = []
    equilibria_count = []

    if normal:
        position_boundaries = np.linspace(-2,2,M+1)
        points_per_position_list = [10 * (norm.cdf(position_boundaries[i+1]) - norm.cdf(position_boundaries[i])) for i in range(M)]
    else:
        points_per_position_list = [points_per_position for _ in range(M)]

    for sim in range(nsim):
        player_positions = {player: np.random.randint(0,M) for player in range(N)}

        # randomly assigning initial positions to each player
        # creating dictionary where key is the position, and value is a list of the players at that position
        player_count = {position: 0 for position in range(M)}
        for player in range(N):
            player_count[player_positions[player]] += 1

        # initialising empty scores dict
        payoffs_per_position = {position: 0 for position in range(M)}

        # looping through each player, calculating thier best response, and moving them to corresponding position
        player_moved = True
        iteration_count = 0
        while player_moved:

            total_score_temp = 0

            player_moved = False

            if random_sequence:
                player_iterable = np.random.randint(0,N,N*10)
            else:
                player_iterable = range(N)

            for player in player_iterable:
                iteration_count += 1
                current_position = player_positions[player]
                player_positions_temp = player_positions.copy()
                player_count_temp = player_count.copy()
                for position in range(M):
                    player_count_temp[player_positions_temp[player]] -= 1
                    player_count_temp[position] += 1
                    player_positions_temp[player] = position
                    payoffs_per_position[position] = payoff_calculation(player, player_positions_temp, player_count_temp, M, points_per_position_list)

                max_payoff = max(payoffs_per_position.values())
                total_score_temp += max_payoff
                best_responses = [k for k,v in payoffs_per_position.items() if v == max_payoff]
                best_response = random.sample(best_responses,1)[0]

                if payoffs_per_position[best_response] > payoffs_per_position[current_position]:
                    player_count[current_position] -= 1
                    player_count[best_response] += 1
                    player_positions[player] = best_response
                    player_moved = True
                    
            if iteration_count > M*100:
                # print(f"No equilibrium found after {M*100} iterations!")
                break
            
        if player_moved == False and player_count not in equilibria_count:
            equilibria_count.append(player_count)
            equilibria.append(list(player_positions.values()))

    if equilibria:
        # plotting histogram
        cm = plt.cm.get_cmap('brg')
        hist_vals = list(itertools.chain.from_iterable(equilibria))
        plt.figure()
        n, bins, patches = plt.hist(hist_vals, bins = M, range=(-0.5,M-0.5), rwidth = 0.8, color='green')
        # To normalize your values
        col = (n-n.min())/(n.max()-n.min())
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        if M > 10:
            plt.xticks(list(range(0,M,5)))
        else:
            plt.xticks(list(range(M)))
        plt.title(f"Equilibria positions for {N} players, {M} strategies", fontsize=15)
        plt.xlabel("Position", fontsize=13)
        plt.show()
        
        return equilibria
    else:
        print(f"No equilibria found after {nsim} simulations!")
        return None
