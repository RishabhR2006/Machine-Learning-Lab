def find_sum_pairs(num_list,target=10):
    total_pairs=0
    matching_pairs=[]
    length=len(num_list)
    for i in range(length):
        for j in range(i+1,length):
            if num_list[i]+num_list[j]==target:
                total_pairs+=1
                matching_pairs.append((num_list[i],num_list[j]))
    return total_pairs,matching_pairs

input_data=[2,7,4,1,3,6]
pair_count,pairs_list=find_sum_pairs(input_data)
print("List:",input_data)
print("Count:",pair_count)
print("Pairs:",pairs_list)