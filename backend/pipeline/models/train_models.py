import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import Ridge
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import joblib

#HLM
df = pd.read_csv(r"C:\Users\a_cas\OneDrive\Documents\Projects\ADME_HLM.csv", sep = '\t')
smiles = df['SMILES']
mols = [Chem.MolFromSmiles(s) for s in smiles]

Y = df['LOG HLM_CLint (mL/min/kg)']

apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize = 512)
fps = []
for m in mols:
    fps.append(apgen.GetFingerprintAsNumPy(m))
X = np.asarray(fps, dtype = float)
print(X.shape)

model = RandomForestRegressor()
model.fit(X, Y)
joblib.dump(model, '../HLM_RFregressor.pickle')


#MDR1 ER
df = pd.read_csv(r"C:\Users\a_cas\OneDrive\Documents\Projects\ADME_MDR1_ER.csv", sep = '\t')
smiles = df['SMILES']
mols = [Chem.MolFromSmiles(s) for s in smiles]

Y = df['LOG MDR1-MDCK ER (B-A/A-B)']
apgen = rdFingerprintGenerator.GetMorganGenerator(radius = 3, fpSize = 1024)
fps = []
for m in mols:
    fps.append(apgen.GetFingerprintAsNumPy(m))
X = np.asarray(fps, dtype = float)
print(X.shape)

model = svm.SVR()
model.fit(X, Y)
joblib.dump(model, '../MDR1_ER_SVR.pickle')


#RLM
df = pd.read_csv(r"C:\Users\a_cas\OneDrive\Documents\Projects\ADME_RLM.csv", sep = '\t')
smiles = df['SMILES']
mols = [Chem.MolFromSmiles(s) for s in smiles]

Y = df['LOG RLM_CLint (mL/min/kg)']
apgen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize = 512, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
fps = []
for m in mols:
    fps.append(apgen.GetFingerprintAsNumPy(m))
X = np.asarray(fps, dtype = float)
print(X.shape)

model = Ridge()
model.fit(X, Y)
joblib.dump(model, '../RLM_ridge.pickle')

