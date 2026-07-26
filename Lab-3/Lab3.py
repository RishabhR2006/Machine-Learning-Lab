import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance as scipy_dist

df=pd.read_excel("marketing_campaign.xlsx")


def custom_label_encode(series,mapping=None):
  if mapping is None:
    unique_vals=list(series.unique())
    mapping={val:idx for idx,val in enumerate(unique_vals)}
  return series.map(mapping),mapping


def custom_one_hot_encode(df_in,column_name):
  unique_vals=df_in[column_name].unique()
  ohe_df=pd.DataFrame()
  for val in unique_vals:
    ohe_df[f"{column_name}_{val}"]=(df_in[column_name]==val).astype(int)
  return ohe_df


def custom_minkowski_distance(v1,v2,p=2):
  v1=np.array(v1,dtype=float)
  v2=np.array(v2,dtype=float)
  return np.sum(np.abs(v1-v2)**p)**(1/p)


df_encoded=df.copy()
edu_order={'Basic':0,'2n Cycle':1,'Graduation':2,'Master':3,'PhD':4}
df_encoded['Education_Encoded'],_=custom_label_encode(
    df_encoded['Education'],edu_order
)

marital_ohe=custom_one_hot_encode(df_encoded,'Marital_Status')

df_transformed=df_encoded.drop(
    columns=['Education','Marital_Status','Dt_Customer']
)
df_transformed=pd.concat([df_transformed,marital_ohe],axis=1)

df_transformed['Income']=df_transformed['Income'].fillna(
    df_transformed['Income'].median()
)

v1=df_transformed.iloc[0].values
v2=df_transformed.iloc[1].values

p_values=list(range(1,11))
distances=[custom_minkowski_distance(v1,v2,p) for p in p_values]
scipy_distances=[scipy_dist.minkowski(v1,v2,p) for p in p_values]

plt.figure(figsize=(8,5))
plt.plot(p_values,distances,'o-b',label='Custom Minkowski Distance')
plt.xlabel('Order Parameter (p)')
plt.ylabel('Distance Value')
plt.title('Minkowski Distance vs Order Parameter (p)')
plt.xticks(p_values)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

comp_df=pd.DataFrame({
    'Order (p)':p_values,
    'Custom Function':distances,
    'SciPy Function':scipy_distances,
    'Absolute Difference':np.abs(
        np.array(distances)-np.array(scipy_distances)
    ),
})
print(comp_df.to_string(index=False))


def custom_dot_product(a,b):
  a=np.array(a,dtype=float)
  b=np.array(b,dtype=float)
  return np.sum(a*b)


def custom_euclidean_norm(a):
  a=np.array(a,dtype=float)
  return np.sqrt(np.sum(a**2))


dot_custom=custom_dot_product(v1,v2)
dot_numpy=np.dot(v1,v2)
norm_v1_custom=custom_euclidean_norm(v1)
norm_v1_numpy=np.linalg.norm(v1)

print(f"Dot Product - Custom: {dot_custom}, NumPy: {dot_numpy}")
print(f"Euclidean Norm - Custom: {norm_v1_custom}, NumPy: {norm_v1_numpy}")


def custom_mean(data):
  data=np.array(data,dtype=float)
  return np.sum(data,axis=0)/len(data)


def custom_variance(data):
  data=np.array(data,dtype=float)
  m=custom_mean(data)
  return np.sum((data-m)**2,axis=0)/len(data)


def custom_std(data):
  return np.sqrt(custom_variance(data))


data_matrix=df_transformed.values
custom_means=custom_mean(data_matrix)
custom_stds=custom_std(data_matrix)

numpy_means=np.mean(data_matrix,axis=0)
numpy_stds=np.std(data_matrix,axis=0)

stat_comp_df=pd.DataFrame({
    'Feature':df_transformed.columns,
    'Custom Mean':custom_means,
    'NumPy Mean':numpy_means,
    'Custom Std':custom_stds,
    'NumPy Std':numpy_stds,
})
print(stat_comp_df.to_string(index=False))

feature_data=df_transformed['Income'].values
hist_counts,hist_bins=np.histogram(feature_data,bins=10)

plt.figure(figsize=(8,5))
plt.hist(feature_data,bins=10,edgecolor='black')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Density Pattern')
plt.grid(True)
plt.tight_layout()
plt.show()

income_mean=custom_mean(feature_data)
income_var=custom_variance(feature_data)
print(f"Income Mean: {income_mean}, Income Variance: {income_var}")


def custom_kmeans(X,k,max_iters=100,random_state=42):
  np.random.seed(random_state)
  X=np.array(X,dtype=float)
  centroids=X[np.random.choice(len(X),size=k,replace=False)]
  for _ in range(max_iters):
    distances=np.array(
        [[custom_minkowski_distance(x,c,p=2) for c in centroids] for x in X]
    )
    labels=np.argmin(distances,axis=1)
    new_centroids=np.array([
        X[labels==i].mean(axis=0) if np.sum(labels==i)>0 else centroids[i]
        for i in range(k)
    ])
    if np.allclose(centroids,new_centroids):
      break
    centroids=new_centroids
  return centroids,labels


k=3
centroids,labels=custom_kmeans(data_matrix,k)
print(f"K-Means completed for K={k}. Centroids shape: {centroids.shape}")