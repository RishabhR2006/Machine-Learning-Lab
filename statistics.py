import random
import statistics

def generate_numbers():
    return [random.randint(1,10) for _ in range(25)]

def get_stats(data):
    mean_val=statistics.mean(data)
    median_val=statistics.median(data)
    try:
        mode_val=statistics.mode(data)
    except statistics.StatisticsError:
        mode_val=statistics.multimode(data)
    return mean_val,median_val,mode_val

random_set=generate_numbers()
avg,mid,most=get_stats(random_set)
print("Numbers:",random_set)
print(f"Mean: {avg:.2f} | Median: {mid} | Mode: {most}")