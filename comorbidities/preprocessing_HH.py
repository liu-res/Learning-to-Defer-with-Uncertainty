import pandas as pd
import numpy as np
import random


def convert_dsfs(df_claims):
    df_claims['DSFS'].fillna(0, inplace=True)
    df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
    df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
    df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
    df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
    df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
    df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
    df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
    df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
    df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
    df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
    df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
    df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12
    return df_claims


def convert_charlson_index(df_claims):
    df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
    df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
    df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 3
    df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 5
    return df_claims


def convert_paydelay(df_claims):
    df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
    return df_claims


def convert_stay(df_claims):
    df_claims['LengthOfStay'].fillna(0, inplace=True)
    df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
    df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
    df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
    df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
    df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
    df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
    df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 7
    df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
    df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
    df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 182
    return df_claims


def convert_age(df_members):
    df_members['AgeAtFirstClaim'].fillna(0, inplace=True)
    df_members.loc[df_members['AgeAtFirstClaim'] == '0-9', 'AgeAtFirstClaim'] = 0
    df_members.loc[df_members['AgeAtFirstClaim'] == '10-19', 'AgeAtFirstClaim'] = 10
    df_members.loc[df_members['AgeAtFirstClaim'] == '20-29', 'AgeAtFirstClaim'] = 20
    df_members.loc[df_members['AgeAtFirstClaim'] == '30-39', 'AgeAtFirstClaim'] = 30
    df_members.loc[df_members['AgeAtFirstClaim'] == '40-49', 'AgeAtFirstClaim'] = 40
    df_members.loc[df_members['AgeAtFirstClaim'] == '50-59', 'AgeAtFirstClaim'] = 50
    df_members.loc[df_members['AgeAtFirstClaim'] == '60-69', 'AgeAtFirstClaim'] = 60
    df_members.loc[df_members['AgeAtFirstClaim'] == '70-79', 'AgeAtFirstClaim'] = 70
    df_members.loc[df_members['AgeAtFirstClaim'] == '80+', 'AgeAtFirstClaim'] = 80
    return df_members

# pre-processing claims data
df_claims = pd.read_csv('Claims.csv')
# remove record with null LengthOfStay that is due to suppression done during the de-identification process
df_claims = df_claims[df_claims.SupLOS == 0]
# remove null records
df_claims = df_claims[~df_claims.isnull().any(axis=1)]
# pre-processing
df_claims = convert_paydelay(df_claims)
df_claims = convert_dsfs(df_claims)
df_claims = convert_charlson_index(df_claims)
df_claims = convert_stay(df_claims)
df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)
df_claims['DSFS'] = df_claims['DSFS'].astype(int)
df_claims['CharlsonIndex'] = df_claims['CharlsonIndex'].astype(int)
df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)
# aggregate by member ID
df_claims_agg = pd.DataFrame()
df_group = df_claims.groupby('MemberID')
df_claims_agg = df_group.agg({
    'ProviderID': 'nunique',
    'Vendor': 'nunique',
    'PCP': 'nunique',
    'PlaceSvc': 'nunique',
    'Specialty': 'nunique',
    'PrimaryConditionGroup': 'nunique',
    'ProcedureGroup': 'nunique',
    'PayDelay': 'sum',
    'CharlsonIndex': 'sum',
    'LengthOfStay': 'sum',
    'DSFS': 'sum',
}).reset_index()
df_claims_agg.columns = ['MemberID', 'Num_Providers', 'Num_Vendors', 'Num_PCPs',
                         'Num_PlaceSvcs', 'Num_Specialities', 'Num_PrimaryConditionGroups', 'Num_ProcedureGroups',
                         'Sum_PayDelay', 'Sum_CharlsonIndex', 'Sum_LengthOfStay', 'Sum_DSFS']
# set lable to 1 if a memebr has any positive Charlson Index, and 0 otherwise
df_claims_agg['Label'] = np.where(df_claims_agg['Sum_CharlsonIndex'] > 0, 1, 0)
df_claims_agg = df_claims_agg.drop('Sum_CharlsonIndex', axis=1)

# pre-processing drug count data
df_drug = pd.read_csv('DrugCount.csv')
df_drug = convert_dsfs(df_drug)
df_drug['DSFS'] = df_drug['DSFS'].astype(int)
df_drug["DrugCount"] = df_drug["DrugCount"].apply(lambda x: int(x.replace("+", "")))
# aggregate by member ID
df_drug_agg = df_drug.groupby(['MemberID'])
df_drug_agg = df_drug_agg.agg({'DrugCount': ['sum'], 'DSFS': ['sum']}).reset_index()
df_drug_agg.columns = ['MemberID', 'DrugCounts', 'Drug_DFDS']

# pre-processing lab data
df_lab = pd.read_csv('LabCount.csv')
df_lab = convert_dsfs(df_lab)
df_lab['DSFS'] = df_lab['DSFS'].astype(int)
df_lab["LabCount"] = df_lab["LabCount"].apply(lambda x: int(x.replace("+", "")))
# aggregate by member ID
df_lab_agg = df_lab.groupby(['MemberID'])
df_lab_agg = df_lab_agg.agg({'LabCount': ['sum'], 'DSFS': ['sum']}).reset_index()
df_lab_agg.columns = ['MemberID', 'LabCounts', 'Lab_DFDS']

# pre-processing members data
df_members = pd.read_csv('Members.csv')
# remove null records
df_members = df_members[~df_members.isnull().any(axis=1)]
df_members = convert_age(df_members)
df_members['AgeAtFirstClaim'] = df_members['AgeAtFirstClaim'].astype(int)
df_members['Sex'] = np.where(df_members['Sex'] == 'M', 1, 0)

print('Num Members in Claims.csv: ', len(df_claims_agg.MemberID.unique()))
print('Num Members in Members.csv: ', len(df_members.MemberID.unique()))
print('Num Members in LabCount.csv: ', len(df_lab_agg.MemberID.unique()))
print('Num Members in DrugCount.csv: ', len(df_drug_agg.MemberID.unique()))

# merge the claims, labs, drugs, and members data:
df_data = df_claims_agg.merge(df_lab_agg, on=['MemberID'], how='left')
df_data = df_data.merge(df_drug_agg, on=['MemberID'], how='left')
df_data = df_data.fillna(0)
df_data = df_data.merge(df_members, on='MemberID', how='left')
df_data = df_data[~df_data.isnull().any(axis=1)]

print('Num Members in merged result: ', len(df_data.MemberID.unique()))
print('original 0/1 ratio: ', df_data.Label.value_counts())
print('downsample fraction: ', df_data.Label.value_counts()[1],' / ', df_data.Label.value_counts()[0])

# down-sample the majority class (Label=0)
fraction = df_data.Label.value_counts()[1] / df_data.Label.value_counts()[0]
df_frac = df_data[df_data.Label == 0].sample(frac=fraction)
df_downsampled = df_frac.append(df_data[df_data.Label == 1], ignore_index=True)

# Shuffle the positive and negative records:
seed = 47
all_idx = list(range(0, df_downsampled.shape[0]))
random.Random(seed).shuffle(all_idx)  # shuffle all_idx inplace
df_input = df_downsampled.iloc[all_idx].reset_index(drop=True)
df_input.to_csv('outputs/df_input.csv', index=False)

print('Num Members in df_input: ', len(df_input.MemberID.unique()))
print('down-sampled 0/1 ratio: ', df_input.Label.value_counts())

# spliting train and test data at 70:30 ratio
random.seed(47)
train_idx = random.sample(list(range(df_input.shape[0])), int(df_input.shape[0]*0.7))
train_data = df_input[df_input.index.isin(train_idx)]
test_data = df_input[~df_input.index.isin(train_idx)]
train_data.to_csv('outputs/df_train.csv', index=False)
test_data.to_csv('outputs/df_test.csv', index=False)
