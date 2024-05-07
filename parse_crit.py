import json;
import os

# 1 Read a time statistic (a mean) from each benchmark directory written by criterion
# 2. Sort the times in increasing order
# 3. Print the (sorted) directory name and times in a form suitable for gnuplot

# Run from ./target/

bench_dirs = (os.listdir("criterion"))

results = []
for _dir in bench_dirs:
    if _dir == 'report':
        continue
    jfile = os.path.join("criterion", _dir, "new", "estimates.json")
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
