### DataFrame의 누락된 값 쉽게 다루기
## impute : 불완전한 값을 산입하다.

from sklearn.impute import SimpleImputer
import pandas as pd

imputer = SimpleImputer(strategy='median')
df = pd.DataFrame()

imputer.fit(df)
print(imputer.statistics_)

X = imputer.transform(df)
new_df = Data(X, columns = df.columns, index = df.index)

