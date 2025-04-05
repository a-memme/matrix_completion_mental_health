import numpy as np
import re as regex

## Functions
def snake_case(s:str):
    # leading/trailing spaces
    s = s.strip()
    # Replace with underscore
    s = regex.sub(r'[\s-]+', '_', s)
    # Convert CamelCase or PascalCase to snake_case
    s = regex.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    s = s.lower()
    # only keep letters/numbers/underscores
    s = regex.sub(r'[^a-z0-9_]', '', s)
    
    return s

#Random data removal
def random_nans(seed, df, perc, exclude_column=None):
    # Random mask where x perc will be set to == True 
    np.random.seed(seed)
    mask = np.random.rand(*df.shape) < perc
    #Exclude column conditionals 
    if exclude_column is not None:
        mask[:, df.columns.get_loc(exclude_column)] = False 
    #Apply mask where mask = NaN
    df_masked = df.mask(mask) 
    
    return df_masked