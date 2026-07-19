import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_average(data_array):
    accumulation=0.0
    for element in data_array:
        accumulation+=element
    return accumulation/len(data_array)

def compute_variance(data_array):
    center=compute_average(data_array)
    squared_deviation=0.0
    for element in data_array:
        squared_deviation+=(element-center)**2
    return squared_deviation/len(data_array)

def evaluate_runtime(metric_callable,dataset,iterations=10):
    intervals=[]
    for _ in range(iterations):
        marker_start=time.perf_counter()
        metric_callable(dataset)
        marker_end=time.perf_counter()
        intervals.append(marker_end-marker_start)
    return np.mean(intervals)

def main():
    source_path="Lab Session Data.xlsx"
    records=pd.read_excel(source_path,sheet_name="IRCTC Stock Price")
    
    records["ConvertedDate"]=pd.to_datetime(records["Date"])
    records["DayOfWeek"]=records["ConvertedDate"].dt.day_name()
    records["MonthIndex"]=records["ConvertedDate"].dt.month
    
    target_series=records["Price"].dropna().to_numpy()
    
    builtin_avg=np.mean(target_series)
    builtin_var=np.var(target_series)
    
    tailored_avg=compute_average(target_series)
    tailored_var=compute_variance(target_series)
    
    duration_builtin_avg=evaluate_runtime(np.mean,target_series)
    duration_tailored_avg=evaluate_runtime(compute_average,target_series)
    duration_builtin_var=evaluate_runtime(np.var,target_series)
    duration_tailored_var=evaluate_runtime(compute_variance,target_series)
    
    wednesday_subset=records[records["DayOfWeek"]=="Wednesday"]
    wednesday_avg=wednesday_subset["Price"].mean()
    
    april_subset=records[records["MonthIndex"]==4]
    april_avg=april_subset["Price"].mean()
    
    records["CleanedChg"]=pd.to_numeric(records["Chg%"].astype(str).str.replace("%","",regex=False),errors="coerce")
    if records["CleanedChg"].isna().all():
        records["CleanedChg"]=records["Chg%"]
        
    negative_filter=lambda value:value<0
    overall_loss_likelihood=np.mean(list(map(negative_filter,records["CleanedChg"].dropna())))
    
    wednesday_cleaned=records[records["DayOfWeek"]=="Wednesday"].dropna(subset=["CleanedChg"])
    wednesday_profit_likelihood=np.mean(wednesday_cleaned["CleanedChg"]>0)
    
    print(f"Package Mean: {builtin_avg}")
    print(f"Package Variance: {builtin_var}")
    print(f"Custom Mean: {tailored_avg}")
    print(f"Custom Variance: {tailored_var}")
    print(f"Mean Difference Acc: {abs(builtin_avg-tailored_avg)}")
    print(f"Variance Difference Acc: {abs(builtin_var-tailored_var)}")
    print(f"Execution Cost - Builtin Mean: {duration_builtin_avg}s")
    print(f"Execution Cost - Custom Mean: {duration_tailored_avg}s")
    print(f"Execution Cost - Builtin Variance: {duration_builtin_var}s")
    print(f"Execution Cost - Custom Variance: {duration_tailored_var}s")
    print(f"Wednesday Sample Average: {wednesday_avg}")
    print(f"April Sample Average: {april_avg}")
    print(f"Probability of Downward Trend: {overall_loss_likelihood}")
    print(f"Conditional Probability of Profit on Wed: {wednesday_profit_likelihood}")
    
    sequence=["Monday","Tuesday","Wednesday","Thursday","Friday"]
    records["DayOfWeek"]=pd.Categorical(records["DayOfWeek"],categories=sequence,ordered=True)
    
    plt.figure(figsize=(7,4))
    plt.scatter(records["DayOfWeek"],records["CleanedChg"],alpha=0.6,edgecolors="none")
    plt.axhline(0,color="black",linestyle=":")
    plt.xlabel("Weekday")
    plt.ylabel("Percentage Change")
    plt.title("Distribution of Daily Change %")
    plt.grid(True,axis="y")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()