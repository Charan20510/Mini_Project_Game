import sys
import os
sys.path.append(os.getcwd())
from Game_Python.bobby_carrot.rl_env import BobbyCarrotEnv

env = BobbyCarrotEnv(map_kind="normal", map_number=4, headless=True)
obs = env.reset()

print("Targets initially:", len(env.target_positions))

# Hack to simulate the crumble collapse manually
env.bobby.coord_src = (6, 10)
env.map_info.data[5 + 10 * 16] = 31

# Without exclude
t1 = env._get_reachable_targets_from((5, 10))
print("Without exclude:", len(t1))

# With exclude
visited = {(5, 10), (6, 10)}
queue = [(5, 10)]
reachable = set()
while queue:
    curr = queue.pop(0)
    if curr in env.target_positions:
        reachable.add(curr)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = curr[0] + dx, curr[1] + dy
        if 0 <= nx < 16 and 0 <= ny < 16 and (nx, ny) not in visited:
            tile = env.map_info.data[nx + ny * 16]
            if tile >= 18 and tile != 30 and tile != 31 and tile != 46:
                visited.add((nx, ny))
                queue.append((nx, ny))

print("With exclude:", len(reachable))

