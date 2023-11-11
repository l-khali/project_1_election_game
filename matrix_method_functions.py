import numpy as np
from scipy.stats import norm

def election_payoff(positions, m):
    """
    Caluclate the payoff for each player given a strategy combination
    positions: array of positions (strategies) chosen by each player
    m: number of possible positions
    
    payoffs: array of payoffs for each player
    """

    # initialise payoffs
    payoffs = np.zeros(len(positions))

    # find unique positions chosen by players
    unique_pos = np.unique(positions)
    
    # for each position, assign votes to the closest players
    for i in range(m):
        # find the minimum distance to an occupied position
        distances = np.array([abs(p-i) for p in unique_pos])
        min_dist = min(distances)
        
        # find the unique positions of players which will earn points
        win_pos = unique_pos[np.where(distances==min_dist)[0]]

        # divide payoff amongst the winning positions
        score = 1/len(win_pos)
        
        # for each winning position, divide points amongst players
        for p in win_pos:
            player_inds = np.where(positions==p)[0]
            payoffs[player_inds] += score/len(player_inds)
    
    return payoffs

def gen_election_mat(n, m, payoff_mat, ind=[]):
    """
    Construct a payoff matrix for a given election game
    n: number of players
    m: number of possible positions
    payoff_mat: n-dimensional empty array to be populated
    ind: list indexing entry in payoff matrix
    
    payoff_mat: n-dimensional array of payoffs
    """
    # end of recursion to start populating payoffs
    if n==1:
        for j in range(m):
            payoff_mat[tuple(ind+[j])] = election_payoff(np.array(ind+[j]),m)
    
    # recursively loop through dimensions
    else:
        n -= 1
        for i in range(m):
            payoff_mat = gen_election_mat(n, m, payoff_mat, ind+[i])
    
    return payoff_mat

def gen_blank_payoff(d, n, m):
    """
    Construct skeleton payoff matrix for a given election game
    d: number of players used to track dimension being constructed
    n: number of players
    m: number of possible positions
    
    returns: n-dimensional empty array of payoffs
    """

    # end of recursion to start constructing empty entries
    if d==1:
        return [[None for k in range(n)] for i in range(m)]
    
    # recursively loop through dimensions
    else:
        return [gen_blank_payoff(d-1, n, m) for j in range(m)]
    
def equilibria(payoffs, m):
    """
    Find equilibria given a payoff matrix
    payoffs: multi-dimensional array of payoffs
    m: number of strategies for each player
    
    eq: set of locations of equilibria in the matrix
    """
    # number of players
    n = np.shape(payoffs)[-1]
    
    # find all indinvidual payoffs for each player
    indv_payoffs = [payoffs[...,i] for i in range(n)]
    
    # find values of best responses
    br = [np.max(indv_payoffs[i],axis=i) for i in range(n)]
    
    # shapes to adjust axes when comapring
    shapes = [[m for i in range(n)] for i in range(n)]
    for i in range(n):
        shapes[i][i] -= (m-1)
    
    # find best responses
    br_loc = [np.argwhere(indv_payoffs[i] == br[i].reshape(shapes[i])) for i in range(n)]
    br_set = [set(map(tuple, br_loc[i])) for i in range(n)]

    # find common best responses which are equilibria
    eq = br_set[0].intersection(br_set[1])
    for i in range(2,n):
        eq = eq.intersection(br_set[i])
    return eq

def election_eq(n, m):
    """
    Find equilibria for a given election game
    n: number of players
    m: number of strategies
    
    unique_eq: array of equilibria
    """
    payoff = gen_election_mat(n, m, np.array(gen_blank_payoff(n, n, m)))
    eq = equilibria(payoff, m)
    sorted_eq = [sorted(np.array(e)) for e in eq]
    unique_eq = np.unique(sorted_eq, axis=0)

    return unique_eq

def norm_election_payoff(positions, m, norm_pos):
    """
    Caluclate the payoff for each player given a strategy combination under a
    normal distribution of voters
    positions: array of positions (strategies) chosen by each player
    m: number of possible positions
    
    payoffs: array of payoffs for each player
    """

    # initialise payoffs
    payoffs = np.zeros(len(positions))

    # find unique positions chosen by players
    unique_pos = np.unique(positions)
    
    # for each position, assign votes to the closest players
    for i in range(m):
        # find the minimum distance to an occupied position
        distances = np.array([abs(p-i) for p in unique_pos])
        min_dist = min(distances)
        
        # find the unique positions of players which will earn points
        win_pos = unique_pos[np.where(distances==min_dist)[0]]

        # divide payoff amongst the winning positions
        score = norm_pos[i]/len(win_pos)
        
        # for each winning position, divide points amongst players
        for p in win_pos:
            player_inds = np.where(positions==p)[0]
            payoffs[player_inds] += score/len(player_inds)
    
    return payoffs

def gen_norm_election_mat(n, m, payoff_mat, norm_pos, ind=[]):
    """
    Construct a payoff matrix for a given election game under a normal
    distribution of voters
    n: number of players
    m: number of possible positions
    payoff_mat: n-dimensional empty array to be populated
    ind: list indexing entry in payoff matrix
    
    payoff_mat: n-dimensional array of payoffs
    """
    # end of recursion to start populating payoffs
    if n==1:
        for j in range(m):
            payoff_mat[tuple(ind+[j])] = norm_election_payoff(np.array(ind+[j]),m, norm_pos)
    
    # recursively loop through dimensions
    else:
        n -= 1
        for i in range(m):
            payoff_mat = gen_norm_election_mat(n, m, payoff_mat, norm_pos, ind+[i])
    
    return payoff_mat

def norm_election_eq(n, m):
    """
    Find equilibria for a given election game under a normal
    distribution of voters
    n: number of players
    m: number of strategies
    
    unique_eq: array of equilibria
    """
    # find normal distribution bounds
    norm_bounds = np.linspace(-2,2,m+1)
    # find available points at each position
    pos_points = [round(norm.cdf(norm_bounds[i+1]) - norm.cdf(norm_bounds[i]), 5) for i in range(m)]

    payoff = gen_norm_election_mat(n, m, np.array(gen_blank_payoff(n, n, m)), pos_points)
    eq = equilibria(payoff, m)
    sorted_eq = [sorted(np.array(e)) for e in eq]
    unique_eq = np.unique(sorted_eq, axis=0)

    return unique_eq

