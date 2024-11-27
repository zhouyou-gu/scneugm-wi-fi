import cProfile
import pstats
import io

def run():
    import time
    time.sleep(1)

# Create a profiler
pr = cProfile.Profile()
pr.enable()  # Start profiling

for i in range(1):
    run()
    
pr.disable()  # Stop profiling

# Create a string stream to capture the profiling data
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)  # Print the top 10 functions

# Output the profiling results
print(s.getvalue())