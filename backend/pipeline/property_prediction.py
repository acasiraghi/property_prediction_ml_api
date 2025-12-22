import joblib
import json
import copy
from pathlib import Path
import pandas as pd 
import numpy as np
from metaflow import FlowSpec, step, Parameter
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize_mol(mol):
    lfc = rdMolStandardize.LargestFragmentChooser()
    mol = lfc.choose(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
    mol = rdMolStandardize.Normalize(mol)
    mol = rdMolStandardize.Reionize(mol)
    Chem.AssignStereochemistry(mol, force = True, cleanIt = True)
    return mol

class PredictFlow(FlowSpec):
    payload_json_string = Parameter(
        'payload_json_string',
        help = 'The json string from the POST request.',
        required = True,
        type = str
    )

    @step
    def start(self):
        self.payload = json.loads(self.payload_json_string)
        self.models = self.payload['config']['models']
        self.data_df = pd.DataFrame(self.payload['data']['rows'])
        self.base_dir = Path(__file__).resolve().parent

        config_path = self.base_dir / 'config' / 'model_configs.json'
        with open(config_path) as f:
            self.model_configs = json.load(f)
        self.supported_models = [c['name'] for c in self.model_configs]
        
        self.next(self.select_models)

    @step
    def select_models(self):
        """Select from available models based on the 'models' parameter."""
        if set(self.models).issubset(set(self.supported_models)):
            self.model_configs = [config for config in self.model_configs if config['name'] in self.models]
        else:
            raise ValueError(f'Unsupported value in --models parameter. Supported models: {self.supported_models}')
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Preprocess input molecules: check validity, standardize structures, deduplicate"""
        processed_smiles = []
        processed_ids = []
        skipped_smiles = []
        
        for smiles_id, smiles in zip(self.data_df['id'], self.data_df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = standardize_mol(mol)
                processed_smiles.append(Chem.MolToSmiles(mol))
                processed_ids.append(smiles_id)
            else:
                skipped_smiles.append(smiles)
                
        self.valid_smiles = processed_smiles
        self.valid_ids = processed_ids
        self.skipped_smiles = skipped_smiles
        self.next(self.featurize, foreach = 'model_configs')

    @step
    def featurize(self):
        """Calculate features."""
        config = copy.deepcopy(self.input)
        if config['featurizer'] == 'atompairs':
            fpgen = rdFingerprintGenerator.GetAtomPairGenerator(**config['featurizer-params'])
        elif config['featurizer'] == 'morgan':
            fpgen = rdFingerprintGenerator.GetMorganGenerator(**config['featurizer-params'])
        elif config['featurizer'] == 'feature-morgan':
            # need to add atom invariant generator here rather than in start step to avoid serialization error
            config['atomInvariantsGenerator'] = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            fpgen = rdFingerprintGenerator.GetMorganGenerator(**config['featurizer-params'])
        mols = [Chem.MolFromSmiles(s) for s in self.valid_smiles]
        self.features = np.asarray([fpgen.GetFingerprintAsNumPy(m) for m in mols])
        self.next(self.predict)

    @step
    def predict(self):
        """Load models and make predictions."""
        self.model_name = self.input['name']
        model_path = self.base_dir / 'models' / f'{self.model_name}.pickle'
        model = joblib.load(model_path)
        self.predictions = model.predict(self.features)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs, include = ['valid_ids', 'valid_smiles'])
        self.results_df = pd.DataFrame({'id': self.valid_ids, 'smiles': self.valid_smiles})

        for inp in inputs:
            # check that number of predictions == number of SMILES
            if len(inp.predictions) != len(self.results_df):
                raise ValueError(f'Number of predictions does not match number of SMILES for {inp.model_name}')
            
            self.results_df[inp.model_name] = inp.predictions
        self.results_json = self.results_df.to_dict(orient = 'records')
        self.next(self.end)

    @step
    def end(self):
        print('Done')

if __name__ == '__main__':
    PredictFlow()