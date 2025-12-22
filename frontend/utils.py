from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

def smiles_to_svg(smiles): 
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    d2d = rdMolDraw2D.MolDraw2DSVG(250, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg_text = d2d.GetDrawingText()
    return svg_text

def svg_to_datauri(svg_text):
    if not isinstance(svg_text, str) or not svg_text.strip():
        return None
    return 'data:image/svg+xml;utf8,' + quote(svg_text)