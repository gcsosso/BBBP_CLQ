# A simple script to draw smiles from CSV with rdkit
import os
import argparse

import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import cairosvg

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()

df = pd.read_csv(args.csv)

#opts = Draw.MolDrawOptions()
#opts.bondLineWidth = 5
#DrawingOptions.bondLineWidth = 5
#DrawingOptions.atomLabelFontSize = 24

for i, row in df.iterrows():
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    #Draw.MolToFile(mol, os.path.join(args.out, f'{smiles}.png'), fitImage=True, size=(100, 100))
    drawer = Draw.MolDraw2DSVG(800, 800)
    drawer.SetLineWidth(10)
    drawer.SetFontSize(40)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    cairosvg.svg2png(bytestring=svg, write_to=os.path.join(args.out, f'{smiles}.png'))
