from map import Map
from terrain import FIELDS, HILL, RIVER
from unit import Infantry, Cavalry, Artillery

# Create map
battlefield = Map(10, 10)
battlefield.set_terrain(3, 3, HILL)
battlefield.set_terrain(4, 3, RIVER)

# Create units
inf = Infantry("3rd Line", 2, 2, quality=2, size=9, morale=6)
cav = Cavalry("Royal Dragoons", 4, 2, quality=5, size=4, morale=9)
art = Artillery("Grand Battery", 1, 1, quality=5, size=5, morale=6)

# Move units (terrain aware)
inf.move(3, 3, battlefield)   # should factor HILL cost
cav.move(4, 3, battlefield)   # should factor RIVER cost

# Combat (terrain modifier automatically applied)
inf.combat(cav, battlefield)
art.bombard(cav)  # Artillery doesn't consider terrain here yet

# Status output
print(inf.status())
print(cav.status())
print(art.status())