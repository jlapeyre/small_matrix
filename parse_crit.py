import json;
import os

bench_dirs = (os.listdir("criterion"))

results = []
for _dir in bench_dirs:
    if _dir == 'report':
        continue
    jfile = os.path.join("criterion", _dir, "new", "estimates.json")
#    print(jfile)
    with open(jfile) as f:
        jcont = json.load(f)
        time = jcont['mean']['point_estimate']
        results.append((time, _dir))

#print(results)
results.sort()
#print()
#print(results)

i = 1
for (time, _dir) in results:
    print('# ', _dir)
    print(i, ' ', time)
    print()
    i += 1

# with open('persons.json') as f:
#    data = json.load(f)
