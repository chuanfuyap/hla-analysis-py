"""Synthetic examples. Run with MPLBACKEND=Agg for headless export.

python examples/visualisation/plot_examples.py --output /path/to/figures
"""
import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import hapy as hp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('hapy_figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent
    assoc = pd.read_csv(base / 'association.csv')
    interactions = pd.read_csv(base / 'interaction.csv')
    hits = [('B', 63, 38175678), ('DPB1', 9, 37289819), ('DPB1', 35, 37289819)]
    for extension in ['png', 'pdf', 'svg']:
        fig, _ = hp.vis.plot_aa_association(
            assoc, x_axis='genomic', log10p_col='META_log10P', highlight_top=True,
            title='Synthetic HLA amino acid position associations',
            position_unit='Mb', figsize=(11, 6),
            save_path=args.output / f'association.{extension}')
        plt.close(fig)
        for kind in ['scatter3d', 'heatmap']:
            fig, _ = hp.vis.plot_aa_interaction(
                interactions, x_axis='genomic', partner_pos_col='TCR_POS', partner_variant_col='PARTNER',
                log10p_col='META_log10P', kind=kind, highlights=hits,
                norm=Normalize(0, 6), position_unit='Mb', figsize=(12, 7),
                annotation_fontsize=8, annotation_offset=(6, 10),
                annotation_offsets={('B', 63, 38175678): (8, 12),
                                    ('DPB1', 9, 37289819): (-230, 45),
                                    ('DPB1', 35, 37289819): (-230, 95)},
                title='Synthetic HLA–variant interactions',
                save_path=args.output / f'interaction_{kind}.{extension}')
            plt.close(fig)


if __name__ == '__main__':
    main()
