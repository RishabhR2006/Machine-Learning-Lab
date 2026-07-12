def determine_range(numbers):
    if len(numbers)<3:
        return "Range determination not possible"
    low=min(numbers)
    high=max(numbers)
    span=high-low
    return f"{span} ({high}-{low})"

dataset=[5,3,8,1,0,4]
range_output=determine_range(dataset)
print("List:",dataset)
print("Result:",range_output)