import gym
import sys
from gym import spaces
from gym.utils import seeding
import numpy as np
from . import settings
#from settings import s, e
s = settings.s
e = settings.e
UP=0
DOWN=1
LEFT=2 
RIGHT=3
BOMB=4
WAIT=5

class Agent(object):
    def __init__(self,id, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.id = id
        self.bombs_left =1
        self.alive =True
        self.events =[]
        self.score =0
    def update_score(self,points):
        self.score = self.score+points
    def make_bomb(self):
        return Bomb((self.x,self.y),self,s.bomb_timer+1,s.bomb_power)
class Item(object):
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]
class Bomb(Item):
    def __init__(self, pos, owner,timer,power):
        super(Bomb,self).__init__(pos)
        self.owner = owner
        self.timer = timer
        self.power = power
        self.active = True
    def get_state(self):
        # return ((self.x, self.y), self.timer, self.power, self.active, self.owner.name)
        return (self.x, self.y, self.timer)
    # arena np array, if is -1 hard
    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x,y)]

        for i in range(1, self.power+1):
            if arena[x+i,y] == -1: break
            blast_coords.append((x+i,y))
        for i in range(1, self.power+1):
            if arena[x-i,y] == -1: break
            blast_coords.append((x-i,y))
        for i in range(1, self.power+1):
            if arena[x,y+i] == -1: break
            blast_coords.append((x,y+i))
        for i in range(1, self.power+1):
            if arena[x,y-i] == -1: break
            blast_coords.append((x,y-i))

        return blast_coords
class Coin(Item):
    def __init__(self, pos):
        super(Coin, self).__init__(pos)
        self.collectable = False
        self.collected = False
    def get_state(self):
        return (self.x, self.y)
class Explosion(object):
    def __init__(self, blast_coords, owner, explosion_timer=s.explosion_timer):
        self.blast_coords = blast_coords
        self.owner = owner
        self.timer = explosion_timer
        self.active = True
#
class Log(object):
    def info(self,message):
        pass
        #print("INFO: "+str(message))
    def debug(self,message):
        pass
        #print("DEBUG: "+str(message))
class BombermanEnv(gym.Env):
    def __init__(self, bombermanrlSettings=s):
        self.screen_height = bombermanrlSettings.rows
        self.screen_width = bombermanrlSettings.cols
        self.action_space = spaces.Discrete(6)# six different actions see above
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.screen_height, self.screen_width), dtype=np.int8)
        self.seed()
        self.logger = Log()
        # Start the first game
        self.reset()
        self.env = self
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def tile_is_free(self, x, y):
        is_free = (self.arena[x,y] == 0)
        if is_free:
            for obstacle in self.bombs + [self.player]:# TODO Players...
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free
    def step(self, action):
        assert self.action_space.contains(action)
        if action == UP and self.tile_is_free(self.player.x, self.player.y - 1):
            self.player.y -= 1
            self.player.events.append(e.MOVED_UP)
        elif action == DOWN and self.tile_is_free(self.player.x, self.player.y + 1):
            self.player.y += 1
            self.player.events.append(e.MOVED_DOWN)
        elif action == LEFT and self.tile_is_free(self.player.x - 1, self.player.y):
            self.player.x -= 1
            self.player.events.append(e.MOVED_LEFT)
        elif action == RIGHT and self.tile_is_free(self.player.x + 1, self.player.y):
            self.player.x += 1
            self.player.events.append(e.MOVED_RIGHT)
        elif action == BOMB and self.player.bombs_left > 0:
            self.logger.info(f'player <{self.player.id}> drops bomb at {(self.player.x, self.player.y)}')
            self.bombs.append(self.player.make_bomb())
            self.player.bombs_left -= 1
            self.player.events.append(e.BOMB_DROPPED)
        elif action == WAIT:
            self.player.events.append(e.WAITED)
        # collect coins
        for coin in self.coins:
            if coin.collectable:
                #for a in self.active_agents:
                a = self.player
                if a.x == coin.x and a.y == coin.y:
                    coin.collectable = False
                    coin.collected = True
                    self.logger.info(f'Agent <{a.id}> picked up coin at {(a.x, a.y)} and receives 1 point')
                    a.update_score(s.reward_coin)
                    a.events.append(e.COIN_COLLECTED)
                    #a.trophies.append(Agent.coin_trophy)
        # simulate bombs and explosion
        #bombs
        for bomb in self.bombs:
            # Explode when timer is finished
            if bomb.timer <= 0:
                self.logger.info(f'Agent <{bomb.owner.id}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                blast_coords = bomb.get_blast_coords(self.arena)
                # Clear crates
                for (x,y) in blast_coords:
                    if self.arena[x,y] == 1:
                        self.arena[x,y] = 0
                        #bomb.owner.events.append(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:# possible bug in GAME engine?==> relive coin
                            if (c.x,c.y) == (x,y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x,y)}')
                                bomb.owner.events.append(e.COIN_FOUND)
                # Create explosion
                self.explosions.append(Explosion(blast_coords, bomb.owner))
                bomb.active = False
                bomb.owner.bombs_left += 1
            # Progress countdown
            else:
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]
        # explosions
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                #for a in self.active_agents:
                a = self.player
                if self.player.alive:
                    if a.alive and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f'Agent <{a.id}> blown up by own bomb')
                            a.events.append(e.KILLED_SELF)
                            #explosion.owner.trophies.append(Agent.suicide_trophy)
                        else:
                            self.logger.info(f'Agent <{a.id}> blown up by agent <{explosion.owner.id}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.reward_kill)
                            explosion.owner.events.append(e.KILLED_OPPONENT)
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False
            # Progress countdown
            explosion.timer -= 1
        a = self.player
        if a in agents_hit:
            a.alive = False
        #    self.active_agents.remove(a)
        #    a.events.append(e.GOT_KILLED)
        #    for aa in self.active_agents:
        #        if aa is not a:
        #            aa.events.append(e.OPPONENT_ELIMINATED)
        #    self.put_down_agent(a)
        self.explosions = [e for e in self.explosions if e.active]
        # check whether coins where collected
        done = self.check_if_all_coins_collected() or self.all_players_dead()
        reward = 0 #0 # TODO coins collected as reward
        if not self.player.alive:
            reward=-100
        if self.check_if_all_coins_collected:
            reward = self.player.score
        return (self._get_obs(), reward, done, {})
    def check_if_all_coins_collected(self):
        return len([c for c in self.coins if not c.collected])==0
    def all_players_dead(self):
        return not self.player.alive
        #return length([a for a in self.players if a])
    # Function that returns the viewed image or so called observation
    # map values: 
    #    2: Coin
    #    -1: WALL
    #    -2: Bomb
    #    -3: Explosion
    #    0 : Free
    #    1 : Crate
    #    3,4,5,6: player
    def _get_obs(self):
        rendered_map = np.copy(self.arena)
        # add coins
        for coin in self.coins:
            if coin.collectable:
                rendered_map[coin.x,coin.y] = 2
        # add bombs
        for bomb in self.bombs:
            rendered_map[bomb.x,bomb.y]=-2
        for explosion in self.explosions:
            for e in explosion.blast_coords:
                rendered_map[e[0],e[1]]=-3
        #TODO add players
        rendered_map[self.player.x,self.player.y]=3
        
        return rendered_map
    def generate_arena(self):
        # Arena with wall and crate layout
        self.arena = (np.random.rand(s.cols, s.rows) < s.crate_density).astype(np.int8)
        self.arena[:1, :] = -1
        self.arena[-1:,:] = -1
        self.arena[:, :1] = -1
        self.arena[:,-1:] = -1
        for x in range(s.cols):
            for y in range(s.rows):
                if (x+1)*(y+1) % 2 == 1:
                    self.arena[x,y] = -1
        # Starting positions
        self.start_positions = [(1,1), (1,s.rows-2), (s.cols-2,1), (s.cols-2,s.rows-2)]
        np.random.shuffle(self.start_positions)
        for (x,y) in self.start_positions:
            for (xx,yy) in [(x,y), (x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if self.arena[xx,yy] == 1:
                    self.arena[xx,yy] = 0
        # Distribute coins evenly
        self.coins = []
        for i in range(3):
            for j in range(3):
                n_crates = (self.arena[1+5*i:6+5*i, 1+5*j:6+5*j] == 1).sum()
                while True:
                    x, y = np.random.randint(1+5*i,6+5*i), np.random.randint(1+5*j,6+5*j)
                    if n_crates == 0 and self.arena[x,y] == 0:
                        self.coins.append(Coin((x,y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x,y] == 1:
                        self.coins.append(Coin((x,y)))
                        break

    def reset(self):
        self.generate_arena()
        self.player = Agent(1,[1,1])
        self.bombs = []
        self.explosions =[]
        return self._get_obs()
    def render(self,mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
    # 2: Coin
    #    -1: WALL
    #    -2: Bomb
    #    -3: Explosion
    #    0 : Free
    #    1 : Crate
    #    3,4,5,6: player
        map = self._get_obs()
        for zeile in map:
            for element in zeile:
                outfile.write("{}".format(["💥","\u1F9E8","❌","👣","❎","🏆","😎"][element+3]))
            outfile.write("\n")
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
if __name__ == "__main__":
    benv = BombermanEnv(s)
    benv.step(RIGHT)
    benv.step(BOMB)
    benv.render()
    benv.step(LEFT)
    benv.render()
    benv.step(WAIT)
    benv.render()
    benv.step(WAIT)
    benv.render()
    benv.step(WAIT)
    benv.render()
    benv.step(WAIT)
    benv.render()