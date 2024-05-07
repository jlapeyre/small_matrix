import json;
import os
import subprocess


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

print('# Generated by parse_crit.py')
print('#')
# git show -s --format="%H" HEAD

git_hash = subprocess.check_output(['git', 'show', '-s', '--format="%H"', 'HEAD'])
print('# git hash:')
print('# ', git_hash.decode('utf-8').strip().replace('"', ''))

git_date = subprocess.check_output(['git', 'show', '--no-patch', '--format="%ci"', 'HEAD'])
print('# git date:', git_date.decode('utf-8').strip().replace('"', ''))
print('#')

i = 1
for (time, _dir) in results:
    print('# ', _dir)
    print(i, ' ', time)
    print()
    i += 1
