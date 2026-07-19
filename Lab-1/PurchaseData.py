import numpy as np

def main():
    X=np.array([
        [20,6,2],
        [16,3,6],
        [27,6,2],
        [19,1,2],
        [24,4,2],
        [22,1,5],
        [15,4,2],
        [18,4,2],
        [21,1,4],
        [16,2,4]
    ])
    y=np.array([386,289,393,110,280,167,271,274,148,198])

    matrix_rank=np.linalg.matrix_rank(X)
    costs=np.linalg.pinv(X)@y

    print("Dimensionality of the vector space: 3")
    print("Number of vectors existing in this vector space: 10")
    print(f"Rank of the feature matrix: {matrix_rank}")
    
    print(f"Cost of Candies: {costs[0]}")
    print(f"Cost of Mangoes: {costs[1]}")
    print(f"Cost of Milk Packets: {costs[2]}")

if __name__=='__main__':
    main()