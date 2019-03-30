from MDP import MDP
import sys


class BellmanDPSolver(object):
	def __init__(self,discount_rate):
		self.mpd = MDP()
		self.actions = self.mpd.A
		self.gamma = discount_rate
		self.policy = {}
		self.current_position = -1
	
	def initVs(self):
		self.values = {(1, 3): 0, (3, 0): 0, (2, 1): 0, (0, 3): 0, (4, 0): 0, (1, 2): 0, (3, 3): 0, (4, 4): 0, (2, 2): 0, (4, 1): 0,
		               (1, 1): 0, 'OUT': 0, (3, 2): 0, (0, 0): 0, (0, 4): 0, (1, 4): 0, (2, 3): 0, (4, 2): 0, (1, 0): 0, (0, 1): 0,
		               'GOAL': 0, (3, 1): 0, (2, 4): 0, (2, 0): 0, (4, 3): 0, (3, 4): 0, (0, 2): 0}
		self.policy = {(1, 3): [], (3, 0): [], (2, 1): [], (0, 3): [], (4, 0): [], (1, 2): [], (3, 3): [], (4, 4): [], (2, 2): [],
		                (4, 1): [], (1, 1): [], 'OUT': [],  (3, 2): [], (0, 0): [], (0, 4): [], (1, 4): [], (2, 3): [], (4, 2): [],
		                (1, 0): [], (0, 1): [], 'GOAL': [], (3, 1): [], (2, 4): [], (2, 0): [], (4, 3): [], (3, 4): [], (0, 2): []}

	
	def BellmanUpdate(self):
		for init_state, value_f in self.values.items():
			max=None
			for action in self.actions:
				temp = 0
				# Transition Table p(s',r | s,a)
				next_states = self.mpd.probNextStates(init_state,action)
				for new_state,prob in next_states.items():
					temp+=prob*(self.mpd.getRewards(init_state,new_state)+self.gamma*self.values[new_state])
				if max == None or temp >= max:
					max = temp
			self.values[init_state]= max
			
			# Greedily compute new policy
			policy_list = []
			max = self.values[init_state]
			for action in self.actions:
				temp = 0
				next_states = self.mpd.probNextStates(init_state,action)
				for new_state,prob in next_states.items():
					temp+=prob*(self.mpd.getRewards(init_state,new_state)+self.gamma*self.values[new_state])
				if temp == max:
					policy_list.append(action)
			self.policy[init_state]= policy_list
			
		return self.values,self.policy
		
		

if __name__ == '__main__':
	discount_rate = 0.9
	solution = BellmanDPSolver(discount_rate)
	solution.initVs()
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

