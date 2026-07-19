import pandas as pd
import numpy as np

def explore_thyroid_dataset(file_path):
    data=pd.read_excel(file_path)
    
    numeric_features=['age','TSH','T3','TT4','T4U','FTI','TBG']
    for col in numeric_features:
        if col in data.columns:
            data[col]=pd.to_numeric(data[col].replace('?',np.nan),errors='coerce')
            
    attributes_metadata={}
    
    for col in data.columns:
        if col=='Record ID':
            continue
            
        missing_count=data[col].isna().sum()+dir(data[col]).count('?') if data[col].dtype==object else data[col].isna().sum()
        
        if col in numeric_features:
            dtype_label='Continuous/Numeric'
            encoding_scheme='None (Continuous Variable)'
            
            min_val=data[col].min()
            max_val=data[col].max()
            val_range=(min_val,max_val)
            
            mean_val=data[col].mean()
            var_val=data[col].var()
            std_val=data[col].std()
            
            q1=data[col].quantile(0.25)
            q3=data[col].quantile(0.75)
            iqr=q3-q1
            outlier_condition=(data[col]<q1-1.5*iqr)|(data[col]>q3+1.5*iqr)
            outliers_found=data[outlier_condition][col].dropna().tolist()
            has_outliers=len(outliers_found)>0
            
        else:
            dtype_label='Nominal/Categorical'
            encoding_scheme='One-Hot Encoding' if col not in ['Condition'] else 'Label Encoding'
            val_range=list(data[col].dropna().unique())
            mean_val,var_val,std_val=None,None,None
            has_outliers=False
            outliers_found=[]
            
        attributes_metadata[col]={
            'datatype':dtype_label,
            'encoding':encoding_scheme,
            'range_or_categories':val_range,
            'missing_values':missing_count,
            'has_outliers':has_outliers,
            'outlier_sample':outliers_found[:5],
            'mean':mean_val,
            'variance':var_val,
            'std_dev':std_val
        }
        
    return attributes_metadata

def main():
    file_name='thyroid_dataset.xlsx'
    analysis_results=explore_thyroid_dataset(file_name)
    
    for feature,metrics in analysis_results.items():
        print(f"Attribute: {feature}")
        print(f"  Type: {metrics['datatype']}")
        print(f"  Suggested Encoding: {metrics['encoding']}")
        print(f"  Missing Value Count: {metrics['missing_values']}")
        print(f"  Range/Distinct Values: {metrics['range_or_categories']}")
        
        if metrics['datatype']=='Continuous/Numeric':
            print(f"  Mean: {metrics['mean']:.4f}")
            print(f"  Variance: {metrics['variance']:.4f}")
            print(f"  Standard Deviation: {metrics['std_dev']:.4f}")
            print(f"  Contains Outliers: {metrics['has_outliers']}")
            if metrics['has_outliers']:
                print(f"  Outliers Sample: {metrics['outlier_sample']}")
        print('-'*40)

if __name__=='__main__':
    main()