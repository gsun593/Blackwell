import numpy as np
import pickle


class Learner:
    def __init__(self, m = 2, gamma = 0.5, delta = 1,):
        self.gamma = gamma
        self.delta = delta
        self.mu = m * 2 - 1
        self.m = m
        self.D = (np.ones((m,m)) - np.identity(m)) / 2
        self.t = 1
    
    def update_regret_step(self, u, a, j, k, pj, pk):
        self.D[j,k] = self.D[j,k] * (self.t-1) / self.t + (u * (pj / pk * (a == k) - (a == j))) / self.t
        
    def update_regret(self, u, a, s):
        for j in range(self.m):
            for k in range(self.m):
                if j != k:
                    pj = s[j]
                    pk = s[k]
                    self.update_regret_step(u, a, j, k, pj, pk)
        self.t += 1
        
    def get_M(self):
        R = np.maximum(self.D, 0)
        return R / self.mu + np.identity(self.m) - np.diag(R.sum(axis = 1)) / self.mu
    
    def get_blackwell(self):
        M = self.get_M()
        eigenValues, eigenVectors = np.linalg.eig(M.transpose())

        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenVectors[:,0] / sum(eigenVectors[:,0])
    
    def get_strategy(self):
        q = self.get_blackwell()
        dt = self.delta / (self.t ** self.gamma)
        return (1 - dt) * q + dt * np.ones(self.m) / self.m
    
    def play_strategy(self):
        return np.random.multinomial(1, self.get_strategy()).argmax()


class Environment:
    def __init__(self, m1 = 2, gamma = 0.5, delta = 1, m2 = 2,
                 utility_mat = np.array([[1,-1], [-1, 1]])):
        if utility_mat.shape != (m1, m2):
            raise Exception('Utility Matrix must match dimension of action space')
        self.learner = Learner(m1, gamma, delta)
        self.m1 = m1
        self.m2 = m2
        self.utility_mat = utility_mat
    
    def play_round(self, a2):
        s1 = self.learner.get_strategy()
        a1 = self.learner.play_strategy()
        u = self.utility_mat[a1, a2]
        self.learner.update_regret(u, a1, s1)
        return(a1, a2, u)


print('How many moves for player 1 (You)?')
while True:
    m2 = input()
    try:
        m2 = int(m2)
        if m2 < 2:
            print('Please input number > 1')
        else:
            break
    except:
        print('Please input number')
        continue
print('How many moves for player 2 (The AI)?')
while True:
    m1 = input()
    try:
        m1 = int(m1)
        if m1 < 2:
            print('Please input number > 1')
        else:
            break
    except:
        print('Please input number')
        continue
        
flat = []
for i in range(m1):
    for j in range(m2):
        print('What is the payoff to your opponent when you play {0} and your oponent plays {1}?'.format(j, i))
        while True:
            u = input()
            try:
                u = float(u)
                break
            except:
                print('Please input number')
                continue
        flat.append(u)

utility_mat = np.array(flat).reshape((m1, m2))
utility_mat = utility_mat / abs(utility_mat).max()
print('Normalizing maximum absolute payoff to 1. This is the payoff matrix for your opponent:')
print(utility_mat)
print('Try to minimize your opponent\'s payoff')
e = Environment(m1, 0.5, 1, m2, utility_mat)
moves = [x for x in range(m2)]
print('Starting game. To stop, enter "exit" at any time')
exit_flag = False
game_record = []
t = 1
sum_util = 0
while True:
    print('Play move in {0}'.format(moves))
    while True:
        a2 = input()
        if a2 == 'exit':
            exit_flag = True
            break
        else:
            try:
                a2 = int(a2)
                if a2 not in moves:
                    print('Please input a valid move')
                else:
                    break
            except:
                print('Please input number')
                continue
    if exit_flag:
        break
    r = e.play_round(a2)
    print('You played: {0}, Opponent played: {1}, Opponent utility: {2},'.format(r[1], r[0], r[2]))
    sum_util += r[2]
    t += 1
    print('Average opponent utility so far: {0}'.format(sum_util / t))
    game_record.append(r)


with open('latest_game.pickle', 'wb') as handle:
    pickle.dump(game_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

