import random
import numpy as np
import scipy.stats as stats
import math

class Tsetlin:
    actions=[0,1] # action 0, action 1
    def __init__(self, N=3, rand_init_dist=random.uniform(0,1)):
        self.N=N
        # random action to initialize it:
        rand_action=np.random.choice([0, 1], p=[rand_init_dist, 1-rand_init_dist])
        # choose random state in that action
        #self.state=random.randint(1,N)+rand_action*N
        self.state=np.random.choice([N, 2*N], p=[0.5, 0.5]) # start state is N or 2N
        self.converged=False
        self.rand_x=random.uniform(0,1)

    def check_converge(self):
    	# check if converged (if its close to end of the chain):
        conv_lim=5 #distance from the end of the 
        if self.state < min(conv_lim, self.N/2): # if state has reached 5 or less
        	self.converged=True
        elif self.state>self.N and (self.state-self.N)<min(conv_lim, self.N/2): # if state has reached N+5 or less
        	self.converged =True
        else:
        	self.converged=False
        
    def reward(self):
        # move 1 --> N if in action 1
        if self.state<=self.N:
            next_state= max(self.state-1,1)
        # move 2N --> N+1 if in action 2
        elif self.state>=(self.N+1):
            next_state=max(self.state-1, self.N+1)
        self.state=next_state

        self.check_converge()

        return next_state
    
    def penalty(self):
        if self.state<self.N and self.state>=1:
            next_state=self.state+1
        elif self.state==self.N:
            next_state=2*self.N
        # move N+1 --> 2N if in action 2
        elif self.state>=(self.N+1) and self.state<(2*self.N):
            next_state=self.state+1
        elif self.state ==(2*self.N):
            next_state=self.N
        self.state=next_state

        self.check_converge()

        return next_state

    
    def get_action(self):
        self.rand_x=random.uniform(0,1) # randomly regen to assess penality (c)
        if self.state<=self.N:
            return self.actions[0]
        else:
            return self.actions[1]
        
    def __repr__(self):
        return "N: "+str(self.N)+", Current State: "+str(self.state)+", Action: "+str(self.get_action())


class Lri:
    actions=[0,1] # action 0, action 1
    def __init__(self, kr=0.9, rand_init_dist=random.uniform(0,1)):
        self.P=[0.5, 0.5]
        #self.P=[rand_init_dist, 1-rand_init_dist]
        self.action=np.random.choice(self.actions, p=self.P)
        self.kr=kr
        self.converged=False
        self.rand_x=random.uniform(0,1)

    def check_converge(self):
    	conv_lim=0.95
    	if any([pi>conv_lim for pi in self.P]):
    		self.converged=True
    	else:
    		self.converged=False
        
    def reward(self):
        #min_val=0.0001
        #P_new=[max(Pi*self.kr, min_val) for Pi in self.P]
        P_new=[Pi*self.kr for Pi in self.P]
        P_new[self.action]=0
        P_new[self.action]=1-sum(P_new)          
        self.P=P_new

        self.check_converge()
        return self.P
    
    def penalty(self):
        return self.P
    
    def get_action(self):
        self.action=np.random.choice(self.actions, p=self.P)
        self.rand_x=random.uniform(0,1) # randomly regen to assess penality (c)
        return self.action
        
    def __repr__(self):
        return "P: "+str(self.P)+", Action: "+str(self.action)+", kr: "+str(self.kr)        

class GooreGame:
    def __init__(self, n_voters=10, theta_opt=0.7, N=3, kr=0.9, automata="Tsetlin", g_mod="unimod"):
        self.theta_opt=theta_opt
        self.n_voters=n_voters
        self.rand_init_dist=random.uniform(0,1)
        # initialize all voters
        if automata=="Tsetlin":
        	self.voters=[Tsetlin(N, rand_init_dist=self.rand_init_dist) for voter in range(n_voters)]
        elif automata=="Lri":
        	self.voters=[Lri(kr=kr, rand_init_dist=self.rand_init_dist) for voter in range(n_voters)]
        else:
        	print("Invalid type of automata, try Tsetlin or Lri.")
        # use voters initial state to init theta
        self.theta=sum(self.get_votes())/self.n_voters 
        self.rewards=self.referee(self.g(self.theta))
        self.p_pen=sum(self.rewards)/len(self.rewards)
        self.g_mod=g_mod
        self.converged = False
        
    def g(self, theta):
        mu = self.theta_opt
        variance = 0.01
        sigma = math.sqrt(variance)
        return 0.9*((stats.norm.pdf(theta, mu, sigma)+1)/(1+stats.norm.pdf(self.theta_opt, mu, sigma)))

    def bimod_g(self, theta):
        #print("bimod")
        mod1=self.g(theta)

        mu = self.theta_opt/2
        variance = 0.01
        sigma = math.sqrt(variance)
        mod2=0.5*((stats.norm.pdf(theta, mu, sigma)+1)/(1+stats.norm.pdf(mu, mu, sigma)))
        return max(mod1, mod2)

    def referee(self, p_rew):
        # return 0s for rewards and 1 for penalties
        p_pen=1-p_rew
        return np.random.binomial(1, p_pen, size=self.n_voters) 
    
    def get_votes(self):
        # get votes from all the voters
        votes=[voter.get_action() for voter in self.voters]
        return votes
    
    #def reward_voters(self, rewards):
    def reward_voters(self):
        for voter_id, voter in enumerate(self.voters):
            #if rewards[voter_id]==1:
            if self.p_pen>voter.rand_x:
                voter.penalty()
            else:
                voter.reward()                
        
    def step(self):
        # get vote
        votes=self.get_votes()
        self.theta= sum(votes)/self.n_voters
        # calculate p_rew
        # if self.g_mod=="unimod":
        # 	p_rew=self.g(self.theta)
        # else:
        # 	p_rew=self.bimod_g(self.theta)
        # # get response from referee
        # self.rewards=self.referee(p_rew)
        # # distribute rewards to voters
        # self.reward_voters(self.rewards) 
        if self.g_mod=="unimod":
            self.p_pen= 1-self.g(self.theta) 
        else:
        	self.p_pen= 1-self.bimod_g(self.theta) 
        self.reward_voters()

        # check each voter if converged  
        self.converged= all([voter.converged for voter in self.voters])

    
    def __repr__(self):
        voters_str=[str(voter) for voter in self.voters]
        s = '\n'
        return s.join(voters_str)        
        #return str(self.voters)
    
    def __str__(self):
        voters_str=[str(voter) for voter in self.voters]
        s = '\n'
        return s.join(voters_str)        
        

def krylov_step(N, state, penalty_probs):
    # choose action
    action=choose_action(N, state)
    # get response from environemnent
    beta= envt_beta(action, penalty_probs)
    # move to next state
    if beta and random.getrandbits(1):
        next_state=penalty(N, state)
    else:
        next_state=reward(N, state)
    return next_state
  

