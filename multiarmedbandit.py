import random
import matplotlib.pyplot as plt
import math
import network

TRAIN_NEW_MODEL = False
DECAY_RATE = 0.99

class Bandit:
    def __init__(self,payout,chance):
        self.payout = payout
        self.chance = chance
    
    def play(self):
        if (random.randint(0,100)/100) <= self.chance:
            return self.payout
        return 0
    

class MAB:
    def __init__(self,num_bandits,max_payout,max_plays):
        self.max_payout = max_payout
        self.num_bandits = num_bandits
        self.bandits = []
        self.k = max_plays

        self.best_bandit = -1
        self.max_exp = -1

        for i in range(num_bandits):
            self.bandits.append(Bandit(random.randint(
                0, max_payout), random.randint(0, 100)))

            if self.bandits[-1].payout*(self.bandits[-1].chance/100) > self.max_exp:
                self.max_exp = self.bandits[-1].payout*(self.bandits[-1].chance/100)
                self.best_bandit = i

    def play(self,i):
        return self.bandits[i].play()

def UCB(reward,n,t):
    return Q(reward,n) + Bound(n,t)

def Bound(n,t):
    if not n:
        return float('inf')
    
    if not t:
        return 0
    
    return ((2*math.log(t,math.e)/n)**0.5)

def Q(reward,n):
    if not n:
        return 0
    
    return reward/n

def select_bandit(tmab,child,rewards,plays,time,log=False):
    mx_score = -1
    best_bandit = -1
    scores = []

    if time < tmab.num_bandits:
        return time

    for i,_ in enumerate(tmab.bandits):
        #print([rewards[i]+1, plays[i]+1, time+1])
        score = child.feed_forward([rewards[i]+1,plays[i]+1,time+1])[0]
        scores.append(score)
        if score > mx_score:
            mx_score = score
            best_bandit = i

    if log: 
        pass
        # print()
        # print(f'child yields the folowing scores {scores}')
    #print(f'selects bandit {best_bandit} with score {mx_score}')
    return best_bandit


def pit(n):
    b = -1
    mx = -1
    for idx,_ in enumerate(n):
        if n[idx] > mx:
            mx = n[idx]
            b = idx
    return b

def train_net(num_trials, evolution_rate, training_bandits, training_plays,max_payout,num_children,num_tests,plot=False):
    parent = network.network([3,5,5,5,5,1])
    convergence = []
    mean_convergence = []

    for trial in range(num_trials):
        print('trial number: ',trial+1,'/',num_trials)

        children = []
        children_scores = [0]*num_children
        
        if not trial:
            rate = 1
        else:
            rate = evolution_rate

        for _ in range(num_children):
            children.append(parent.birth(rate))

        # maintain un-evolved child
        children[random.randint(0,len(children)-1)] = parent.birth(rate/100)

        for test in range(num_tests):
            print('test number: ',test+1,'/',num_tests)

            tmab = MAB(training_bandits, max_payout, training_plays)
            children_fitness = [0]*num_children

            for i,child in enumerate(children):
                rewards = [0]*training_bandits
                plays = [0]*training_bandits

                for time in range(training_plays):
                    if test == num_tests-1 and time == training_plays-1:
                        bandit = select_bandit(tmab,child,rewards,plays,time,True)
                    else:
                        bandit = select_bandit(
                            tmab, child, rewards, plays, time)
                        
                    rew = tmab.play(bandit)
                    rewards[bandit] += rew
                    plays[bandit] += 1
                    children_fitness[i] += rew

                #print(f'child {i} in test {test} gets the following rewards from bandits {rewards}')
                
            best_child_idx = pit(children_fitness)
            children_scores[best_child_idx] += 1
            print('test',test,'completed with net',best_child_idx+1,'as the winner.')
            print('net received',children_fitness[best_child_idx],'fitness during this test')
            print('scores:',children_scores)
            print()
            print('children fitness during test:',children_fitness)
            convergence.append(children_fitness[best_child_idx])
            mean_convergence.append(sum(convergence)/len(convergence))
        
        # decay evolution
        evolution_rate *= DECAY_RATE

        best_child_idx = pit(children_scores)
        best_child = children[best_child_idx]
        parent = best_child

    print('training complete')
    parent.save('./network_save.txt')
    print('model saved')

    if plot:   
        plt.plot([x for x in range(len(convergence))],convergence)
        plt.plot([x for x in range(len(convergence))], mean_convergence,'r--')
        plt.show()

    return parent
                    

def main():
    num_bandits,max_payout,max_plays = 50,100,500

    mab1 = MAB(num_bandits,max_payout,max_plays)

    # collect payout of perfect play
    y = 0
    ys = []
    # collect payout of random play
    payout_random = 0
    randoms = []
    # collect payout of ucb play
    payout_epsilon_greedy_decay = 0
    egds = []
    epsilon = 0
    decay_rate = (max_plays-1)/max_plays
    rewards = [0]*num_bandits
    plays = [0]*num_bandits
    # collect payout of neural net strat
    num_generations = 100
    evolution_rate = 0.35
    training_bandits = 50
    training_plays = training_bandits*10
    generation_size = 10
    num_tests = 3

    if TRAIN_NEW_MODEL:
        net = train_net(num_generations,evolution_rate,training_bandits,training_plays,max_payout,generation_size,num_tests)
    else:
        print('loaded network')
        net = network.network([1,1]).load('./network_save.txt')

    rewards = [0]*num_bandits
    plays = [0]*num_bandits
    nets = []
    net_payout = 0

    print('playing',max_plays,'times')

    for i in range(max_plays):
        # playing randomly
        payout_random += mab1.play(random.randint(0,num_bandits-1))
        randoms.append(payout_random)
        # playing perfectly
        y += mab1.play(mab1.best_bandit)
        ys.append(y)

        # epsilon greedy decay
        j = 0
        # ucb 
        if True:
          eg = -1
          # make greedy choice
          for g in range(num_bandits):
              e = UCB(rewards[g],plays[g],i)
              if e > eg:
                  eg = e
                  j = g

        rew = mab1.play(j)
        rewards[j] += rew
        payout_epsilon_greedy_decay += rew
        plays[j] += 1
        egds.append(payout_epsilon_greedy_decay)
        epsilon *= decay_rate

        choice = select_bandit(mab1,net,rewards,plays,i)
        rew = mab1.play(choice)
        net_payout  += rew
        nets.append(net_payout)
        rewards[choice] += rew
        plays[choice] += 1
        

    plt.plot([x for x in range(max_plays)],randoms,'r--',label='random strategy')
    plt.plot([x for x in range(max_plays)], egds, 'y--', label='ucb')
    plt.plot([x for x in range(max_plays)],ys,'b--',label='perfect play')
    plt.plot([x for x in range(max_plays)], nets, 'g--', label='neural net')
    
    print(mab1.bandits[mab1.best_bandit].payout,(mab1.bandits[mab1.best_bandit].chance/100),mab1.max_exp,mab1.best_bandit)
    plt.show()

main()