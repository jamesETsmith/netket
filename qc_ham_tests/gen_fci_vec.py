import numpy as np
from itertools import combinations

nelec = 4
norb = nelec * 2
loc = [0, 1]

confs = []

# for o1 in loc:
#     for o2 in loc:
#         for o3 in loc:
#             for o4 in loc:
#                 for o5 in loc:
#                     for o6 in loc:
#                         for o7 in loc:
#                             for o8 in loc:
#                                 conf = np.array([o1, o2, o3, o4, o5, o6, o7, o8])
#                                 if np.sum(conf) != nelec:
#                                     break
#                                 else:
#                                     occ = np.where(conf == 1)[0]
#                                     sz = 0
#                                     for o in occ:
#                                         if o % 2 == 0:
#                                             sz += 1
#                                         elif o % 2 == 1:
#                                             sz -= 1

#                                     if sz == 0:
#                                         confs.append(occ)
acombs = list(combinations(range(nelec), int(nelec / 2)))
bcombs = list(combinations(range(nelec), int(nelec / 2)))
print(acombs)
print(bcombs)

for a in acombs:
    for b in bcombs:
        conf = [a[0] * 2, b[0] * 2 + 1, a[1] * 2, b[1] * 2 + 1]
        conf.sort()
        confs.append(conf)

#
# print(confs)
for conf in confs:
    print(conf)
print(len(confs))

nk_confs = [
    [0, 1, 2, 3],
    [0, 1, 2, 5],
    [0, 1, 2, 7],
    [0, 1, 3, 4],
    [0, 1, 3, 6],
    [0, 1, 4, 5],
    [0, 1, 4, 7],
    [0, 1, 5, 6],
    [0, 1, 6, 7],
    [0, 2, 3, 5],
    [0, 2, 3, 7],
    [0, 2, 5, 7],
    [0, 3, 4, 5],
    [0, 3, 4, 7],
    [0, 3, 5, 6],
    [0, 3, 6, 7],
    [0, 4, 5, 7],
    [0, 5, 6, 7],
    [1, 2, 3, 4],
    [1, 2, 3, 6],
    [1, 2, 4, 5],
    [1, 2, 4, 7],
    [1, 2, 5, 6],
    [1, 2, 6, 7],
    [1, 3, 4, 6],
    [1, 4, 5, 6],
    [1, 4, 6, 7],
    [2, 3, 4, 5],
    [2, 3, 4, 7],
    [2, 3, 5, 6],
    [2, 3, 6, 7],
    [2, 4, 5, 7],
    [2, 5, 6, 7],
    [3, 4, 5, 6],
    [3, 4, 6, 7],
    [4, 5, 6, 7],
]

not_found = 0
for nkc in nk_confs:
    nkc.sort()
    if nkc not in confs:
        print("NOT FOUND IN FCI CONFS", nkc)
        not_found += 1

print("Not Found", not_found)
