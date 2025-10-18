## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

### "Encoding.csv"

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="355" height="344" alt="op1 1" src="https://github.com/user-attachments/assets/25b52e22-eb40-4fc1-82d2-d9b9708ee280" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="418" height="220" alt="op1 2" src="https://github.com/user-attachments/assets/5683df37-06e7-46c0-afb6-bf2944458948" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="348" height="327" alt="op1 3" src="https://github.com/user-attachments/assets/34ef2641-e0bb-4689-8dd2-b8cd5208864e" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="333" height="356" alt="op1 4" src="https://github.com/user-attachments/assets/e8279003-390d-4012-a1a1-3322ffe84aa3" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="682" height="387" alt="op1 5" src="https://github.com/user-attachments/assets/dfd0ba32-3048-47d6-adf0-16e8b906b418" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="600" height="315" alt="op1 6" src="https://github.com/user-attachments/assets/776f9904-09c7-4f18-88e7-9ef48108341c" />

### "data.csv"

```
pip install --upgrade category_encoders
```
<img width="998" height="298" alt="op1 7" src="https://github.com/user-attachments/assets/a396784a-f7e7-4621-95e3-8e1c0e1337bd" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="502" height="350" alt="op1 8" src="https://github.com/user-attachments/assets/397845f7-3745-48a8-95da-95dc1ed32e6e" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="607" height="357" alt="op1 9" src="https://github.com/user-attachments/assets/a07f7ce4-3ff2-44b9-9284-dfde7a452fe0" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="501" height="385" alt="op1 10" src="https://github.com/user-attachments/assets/e7c1e17a-3800-46ab-ba90-54c9c8d5a771" />


### "Data_to_Transform.csv"

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="675" height="418" alt="op1 11" src="https://github.com/user-attachments/assets/defa6932-30e2-4074-a972-33683f251d90" />

```
df.skew()
```
<img width="272" height="188" alt="op1 12" src="https://github.com/user-attachments/assets/a6c705ad-dae8-46c9-b2b7-c27939fce1d0" />

```
np.log(df["Highly Positive Skew"])
```
<img width="294" height="391" alt="op1 13" src="https://github.com/user-attachments/assets/9a724aca-4047-4ab2-ae84-852249ff19e8" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="396" height="391" alt="op1 14" src="https://github.com/user-attachments/assets/a8527fec-0a84-4862-b8c8-66ee0646cbc6" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="320" height="393" alt="op1 15" src="https://github.com/user-attachments/assets/873c5109-5c17-4a76-8059-339cd53ac351" />

```
np.square(df["Highly Positive Skew"])
```
<img width="304" height="387" alt="op1 16" src="https://github.com/user-attachments/assets/a9954079-d65b-4a36-933a-fb6504ad5503" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="865" height="375" alt="op1 17" src="https://github.com/user-attachments/assets/3ff07ecd-c3c9-462c-b796-1b950b145645" />

```
df.skew()
```
<img width="307" height="215" alt="op1 18" src="https://github.com/user-attachments/assets/a778e5fd-0600-4a36-9bd8-e762aed5edf1" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="639" height="256" alt="op1 19" src="https://github.com/user-attachments/assets/7d06dd32-ba7f-4a29-a131-329e9dd79889" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1217" height="408" alt="op1 20" src="https://github.com/user-attachments/assets/1371db95-6c70-4827-a0ba-d85d24f38c5d" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="580" height="442" alt="op1 21" src="https://github.com/user-attachments/assets/12ff6140-babd-42d9-a937-393defb77a71" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="553" height="393" alt="op1 22" src="https://github.com/user-attachments/assets/1ea25db5-c8be-47e0-92c6-395681651ea5" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="616" height="439" alt="op1 23" src="https://github.com/user-attachments/assets/97af2b62-e4db-497f-b1f0-f45177c04583" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="553" height="410" alt="op1 24" src="https://github.com/user-attachments/assets/718d9e07-3b19-474f-9f31-836044d24fd9" />

# RESULT:
Therefore all the python program codes are executed and verified successfully.
