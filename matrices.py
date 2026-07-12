def multiply_sq_matrices(mat_x,mat_y):
    dim=len(mat_x)
    prod=[[0]*dim for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                prod[i][j]+=mat_x[i][k]*mat_y[k][j]
    return prod

def compute_matrix_power(matrix,p):
    dim=len(matrix)
    result=[[1 if i==j else 0 for j in range(dim)] for i in range(dim)]
    for _ in range(p):
        result=multiply_sq_matrices(result,matrix)
    return result

a_matrix=[[2,3],[1,4]]
exponent=3
powered_matrix=compute_matrix_power(a_matrix,exponent)
print("Matrix A:",a_matrix)
print(f"A^{exponent}:")
for row in powered_matrix:
    print(row)