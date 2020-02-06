import pandas as pd 
import numpy as np
import cvxpy as cp
from heapq import nlargest

def position_indexes(all_pos, all_points, df, idx, position):
    '''
    finding indexes in aggregate data of each player
    in a particular position
    '''
    homes = []
    for idx_ in all_points[player_constraint(position, df, idx)]:
        homes.append(all_pos.index(idx_))
    return homes


def player_constraint(position, df, idx ):
    '''
    this function actually finds all players given a certain postition and their indexes
    (for scoring) as idx. This _technically_ forms a constraint, but not directly like
    the name here may imply. Is this a bad naming convention? Yes. Should I change it? Also yes.
    Will I? Probably not. 
    '''
    df_chose = df.groupby('name').first()
    df_chose = df_chose[df_chose.role.str.contains(position)]
    return df_chose.index.tolist()

def optimize_choice(players, scores, df, g, taken, mine): 
    ohboy = df[df.name.isin(list(scores.iloc[:,players]))]
    playerids = ohboy.groupby('name').count().index
    
    test = scores.mean()
    R = np.array(scores[playerids].mean())
    Q = np.array(scores[playerids].cov())
    x = cp.Variable(len(players))
    
    gamma = cp.Parameter(nonneg=True)
    ret = R.T * x 
    risk = cp.quad_form(x, Q)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk), 
                   [cp.sum(x) == 1, 
                    x >= 0])
    
    gamma.value = g
    prob.solve()
    indexes = list(range(len(players)))
    # Find the index of the player who has the largest proportion of 
    # investment, provided they haven't already been selected 
    # then find the next largest 
    largest = nlargest(len(players), indexes, key=lambda i: x.value[i])
    for i in range(len(largest)):
        pot = players[largest[i]]
       # print(pot, taken)
        if pot in taken:
            continue
        if pot in mine:
            continue
        else: 
            mine.append(pot)
            taken.append(pot)
            return mine, taken, pot

def optim_player(scores, 
                      taken, 
                      mine, 
                      gammaa, 
                      df,
                      offence,
                      tank,
                      support,
                      sub_gamma = .5, 
                      selection = "max",
                      max_salary = False,
                      team_size = 6,
                      min_t = 2,
                      min_o = 2,
                      min_s = 2,
                      full_team = False):
    '''
    This function solves the binary linear programming problem 
    max(r^T x - gamma x^T Q x) where r is the average score per game for a player,
    X is a binary player vector, gamma is the "risk tolerance" parameter, and Q is the 
    covariance matrix of all the scores for each palyer. This is also subject 
    to certain constraints such as maximum salary, player numbers, and number of players
    in a given postion.
    '''
  
    x = cp.Variable(len(scores.mean()),boolean = True)
    
    gamma = cp.Parameter(nonneg=True)
    ret = np.array(scores.mean()).T * x
   
    sigma = np.array(scores.cov())
    risk = cp.quad_form(x, sigma)
    
    constraints = []
    # Cannot pick taken players
    for i in range(len(taken)):
        if taken[i] not in mine:
            constraints.append(x[taken[i]] == 0)
    # Must pick players we already have chosen 
    for i in range(len(mine)):
        constraints.append(x[mine[i]] == 1)
    # Add the salary constraint if we need to 
    if max_salary:
        S = np.diag(df.groupby('player_id').max().Salary.tolist())/10000000
         # L1 norm here, absolute value is fine as no salaries should be negative.
        constraints.append(cp.norm(S @ x, p=1) <= max_salary)
    
    constraints = constraints + [cp.sum(x) == team_size,
                   cp.sum(x[offence]) == min_o,
                   cp.sum(x[tank]) == min_t,
                   cp.sum(x[support]) == min_s] 
                
    # actually defining our problem 
    prob = cp.Problem(cp.Maximize(ret - gamma*risk),
                   constraints)
   
    gamma.value = gammaa
    # TODO: we can probably tighten some of these up 
    # Note: after this x is defined as our players, 
    prob.solve(parallel=True,   
               mi_max_iters=500,
               mi_abs_eps = 1e-5,
               mi_rel_eps = 1e-2,
               max_iters=200,
               abstol = 1e-6,
               reltol = 1e-5,
               feastol = 1e-6,
               abstol_inacc = 5e-4,
               reltol_inacc = 5e-4,
               feastol_inacc = 1e-3)
    
    # Pick highest score player
    # TODO: Update picking stradegy to also include as option sqrt(scores.mean**2 + scores.std**2)
    # Also add _another_ optimization problem with just the team and pick the one with the highest
    # investment proportion (no longer binary - but just with the players we have chosen)
    
    # finding which indexes are non zero (within floating point)
    players = list(np.where(x.value.round(1) ==1)[0])
    new_players = [x for x in players if x not in mine]
    if full_team:
        risk_data = cp.sqrt(risk).value
        return_data = ret.value
        return players, risk_data, return_data

    if selection == 'max':
       
        possible = np.take(np.array(scores.mean()), new_players)
        to_take = list(scores.mean()).index(max(possible))

        mine.append(to_take)
        taken.append(to_take)
         # print('max', to_take)
        return mine, taken, to_take
    
    if selection == 'rms':
        mean_ = scores.mean()
        std_ = scores.std()
        values_ = np.sqrt(mean_**2 + std_**2)
        possible = np.take(values_, new_players)
        to_take = list(values_).index(max(possible))
        #print(to_take)
       # print(to_take, "rms")
        mine.append(to_take)
        taken.append(to_take)
        
        return mine, taken, to_take

    if selection == 'optim':
         mine, taken, to_take = optimize_choice(players, scores, df, sub_gamma, taken, mine)
         #print(to_take, 'optim')
         return mine, taken, to_take


def input_name():
    while True:
        name = input("Please enter player name ")
        
        return name
       
            
            
def human(df_, all_points, name, taken, mine):
    '''
    a function for manual entry and seletion of players 
    if competing against people
    '''
    # in case there's a new player not in the optimization 
    
    while True:
        df = df_[(df_.name.str.contains(name, case=False))]
        
        if len(df.game_id) == 0:
            print("empty data frame?", name)
            if name == "ROOKIE OVERRIDE":
                return mine, taken
            else:
                print("empty data frame? spelling mistake most likely")
                name = input_name()
            continue
        else: 
            p = df['name'].unique()[0]
            df2 = all_points.mean().reset_index()
            player_index = list(df2[df2['name'] == p].index)[0]
            if player_index in taken:
                print('player alread taken, try another')
                name = input_name()
                continue
            mine.append(player_index)
            taken.append(player_index)
            break
    print(mine, taken)
    return mine, taken

def draft(functions, order, team_size=17, pause = False, team_names = None,  **kwargs):
    '''  
    This function is to run a draft which decides on a team. The 'functions' argument
    is a list of functions (defined above) which can be used to simulate players, and 
    order is the order in which those functions (players) will draft. Note that 
    this order is automatically reversed during the draft process. 
    '''

    greedy_selections = kwargs['greedy_selections']
    taken = []
    # the teams
    mine = [[] for i in range(len(functions))]
    df = kwargs['df']
    all_points = kwargs['scores']
    for i in range(team_size):
        print("Beginning round", i)
        for j in order:
            #print(j)
            if team_names:
                print(team_names[j])
            if functions[j].__name__ == 'optim_player':
                mine[j], taken, to_take = functions[j](scores=kwargs['scores'],
                                                       df=kwargs['df'],
                                                       taken=taken, 
                                                       mine=mine[j],
                                                       gammaa=kwargs['gammaa'][j],
                                                       offence=kwargs['offence'],
                                                       tank=kwargs['tank'],
                                                       support=kwargs['support'],
                                                       selection=kwargs['selection'][j],
                                                       sub_gamma=kwargs['sub_gamma'][j])
                print("Optim Player order ", j, " with")
                print("Gamma = ", kwargs['gammaa'][j], "selection = ", kwargs['selection'][j])
                
                ohboy = df[df.name.isin(list(all_points.iloc[:,[to_take]]))]
                playerids = ohboy.groupby('name').count().index
                print(playerids)
                n = df[df.name.isin(playerids)]['name'].drop_duplicates().values
                print(n)
                print("Chose player: ", n)
                print()
                if pause:
                    input("Press enter to continue")

    
            if functions[j].__name__ == 'human':
                name = input_name()
                mine[j], taken = human(df_ = kwargs['df'],
                                       all_points = kwargs['scores'], 
                                       name=name, 
                                       taken=taken, 
                                       mine=mine[j])


            
                
        order = order[::-1]
    # gotta unwrap the dictionary 
    # if greedy_selections: 
    # for key in greedy_selections:
    #     mine[index_of_greed] += greedy_selections[key]
    
    return taken, mine