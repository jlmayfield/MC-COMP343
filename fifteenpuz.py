import copy,random, heapq, sys


class Node:
  """
  Search Tree Node
  """
  def __init__(self, state, action=None, parent=None, cost=0, depth=0):
    self.state = state
    self.action = action
    self.parent = parent
    self.cost = cost
    self.depth = depth

  def __repr__(self):
    return f"Node{hex(id(self))}({self.state},{self.action},{self.cost},{self.depth},{hex(id(self.parent))})"


class Frontier:
  """
  Frontier (priority queue) for Best-First Search
  """
  def __init__(self, f):
    """
    f is the eval function which maps Node -> Num
    """
    self.f = f
    self._count = ~sys.maxsize
    self._queue = []

  def enqueue(self, node):    
    item = (self.f(node), self._count, node)
    self._count += 1
    heapq.heappush(self._queue, item)

  def pop(self):
    if len(self._queue) > 0:
      item = heapq.heappop(self._queue)
      return item[2]

  def isempty(self):
    return len(self._queue) == 0

  def size(self):
    return len(self._queue)


class FifteenPuzzleEnv:

  def __init__(self):
    self.init = FifteenPuzzleEnv._randomstart()

  def _parity(st):
    flat = [e for l in st for e in l]
    n_inv = sum([len(list(filter(lambda n: n < flat[i],flat[i+1:]))) for i in range(15)])
    return n_inv % 2
    
  def _randomstart():
    g = list(range(1,16))
    random.shuffle(g)
    n_inv = sum([len(list(filter(lambda n: n < g[i],g[i+1:]))) for i in range(15)])
    if n_inv % 2 == 0:
      brow = random.choice([1,3])
      bcol = random.randrange(0,4)
      bidx = brow*4 + bcol
      g.insert(bidx,0)
    else:
      brow = random.choice([0,2])
      bcol = random.randrange(0,4)
      bidx = brow*4 + bcol
      g.insert(bidx,0)   
    g = [g[i:i+4] for i in range(0,16,4)]
    return g
    
  def isGoal(st):
    return st == [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 0]] 

  def initialState(self):
    return self.init

  def getActions(st):
    brow = [0 in l for l in st].index(True)
    bcol = st[brow].index(0)

    acts = []
    if brow > 0:
      acts.append('up')
    if brow < 3:
      acts.append('down')
    if bcol > 0:
      acts.append('left')
    if bcol < 3:
      acts.append('right')
    return acts

  def applyAction(act, st):
    st = copy.deepcopy(st)
    brow = [0 in l for l in st].index(True)
    bcol = st[brow].index(0)
    if act == 'left':
      st[brow][bcol], st[brow][bcol - 1] = \
      st[brow][bcol - 1], st[brow][bcol]
    elif act == 'right':
      st[brow][bcol], st[brow][bcol + 1] = \
      st[brow][bcol + 1], st[brow][bcol]
    elif act == 'up':
      st[brow][bcol], st[brow - 1][bcol] = \
      st[brow - 1][bcol], st[brow][bcol]
    elif act == 'down':
      st[brow][bcol], st[brow + 1][bcol] = \
      st[brow + 1][bcol], st[brow][bcol]
    return st

def expand(st):
  act = FifteenPuzzleEnv.getActions(st)
  for a in act:
    yield (a,FifteenPuzzleEnv.applyAction(a,st))

def extractSolution(node):
  path = []
  while node.parent != None:
    path.append((node.action,node.state))
  path.reverse()
  return path

## Some Eval functions and Eval Function Factories 
def bfseval(node):
  return node.depth  

def dfseval(node):
  return -node.depth

def greedyEval(h):
  return lambda node: h(node.state)

def aStarEval(h):
  return lambda node: node.cost + h(node.state)

def weightedAStarEval(h,w):
  return lambda node: node.cost + w*h(node.state)

def misplacedtiles(st):
  flat = [e for l in st for e in l]
  return sum([0 if flat[i] == i+1 else 1 for i in range(15)])

def mdist(st):
  '''
  Expected row of n>0 is (n-1) // 4
  Expected col of n>0 is (n-1) % 4
  dist = abs(r-E[r]) + abs(c-E[c])
  '''
  def tiledist(n,r,c):
    return abs(r-(n-1) // 4) + abs(c-(n-1)%4)
  dists = [[tiledist(st[r][c],r,c) if st[r][c] != 0 else 0 for r in range(4)]  for c in range(4)]
  return sum([e for l in dists for e in l])
  
  

def bestfirstsearch(problem,f):
  numexpanded = 1
  maxfrontier = 1
  front = Frontier(f)
  reached = {} #state -> node
  curr = Node(problem.initialState())
  front.enqueue(curr)
  reached[str(curr.state)] = curr
  while not front.isempty():
    curr = front.pop()    
    if FifteenPuzzleEnv.isGoal(curr.state):
      return curr,numexpanded,maxfrontier    
    for a,st in expand(curr.state):
      cost = curr.cost + 1
      if str(st) not in reached or reached[str(st)].cost > cost:
        nxt = Node(st,a,curr,cost,curr.depth+1)
        reached[str(st)] = nxt
        front.enqueue(nxt)
    maxfrontier = max(maxfrontier,front.size())
    numexpanded += 1
    if numexpanded % 10000 == 0:
      print('expanded',numexpanded)
      print('curr cost',curr.cost)
      print('frontier size', front.size())
      
  return None
    

p = FifteenPuzzleEnv()
print(p.initialState())
#bfs = bestfirstsearch(p,bfseval)
h1 = bestfirstsearch(p,aStarEval(mdist))  
  
  

