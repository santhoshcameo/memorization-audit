"""
Regenerate Figure 4 for the npj Digital Medicine submission.

Panel a: High-memorization MIA risk bars (existing finding, unchanged).
Panel b: Multi-seed Spearman rho across ImageNet (R/V), MedSAM, BiomedCLIP for
         HAM10000, ChestX-ray, ODIR-5K, Kvasir (all four imbalanced datasets). Replaces the
         OLD single-seed "MedSAM reverses" panel that was an artifact of an
         undertrained MedSAM run + asymmetric scoring protocol.
"""
from pathlib import Path
import warnings
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ROOT = Path('.')
OUT_JPG = ROOT / 'figures' / 'fig4_mia_medsam.jpg'
OUT_PDF = ROOT / 'figures' / 'fig4_mia_medsam.pdf'

# ---- Panel a data (existing claim, "31--99 pp more attackable") -----------
PANEL_A = [
    ('HAM\nViT',     0.95, 0.33),  # 62 pp diff
    ('CXR\nResNet',  0.99, 0.14),  # 85 pp
    ('CXR\nMAE',     0.99, 0.00),  # 99 pp
    ('Kvasir\nMAE',  0.93, 0.00),  # 93 pp
]


def compute_panel_b():
    """Multi-seed Spearman rho per (dataset, architecture)."""
    configs = [
        ('HAM10000',   'ham1000_baseline',           'resnet50',   'ImageNet ResNet-50'),
        ('HAM10000',   'ham1000_baseline',           'vit',        'ImageNet ViT'),
        ('HAM10000',   'ham1000_baseline_medsam_v2', 'medsam',     'MedSAM'),
        ('HAM10000',   'ham1000_baseline_biomedclip','biomedclip', 'BiomedCLIP'),
        ('ChestX-ray', 'chestxray_baseline',           'resnet50',   'ImageNet ResNet-50'),
        ('ChestX-ray', 'chestxray_baseline',           'vit',        'ImageNet ViT'),
        ('ChestX-ray', 'chestxray_baseline_medsam_v2', 'medsam',     'MedSAM'),
        ('ChestX-ray', 'chestxray_baseline_biomedclip','biomedclip', 'BiomedCLIP'),
        ('ODIR-5K',    'odir5k_baseline',           'resnet50',   'ImageNet ResNet-50'),
        ('ODIR-5K',    'odir5k_baseline',           'vit',        'ImageNet ViT'),
        ('ODIR-5K',    'odir5k_baseline_medsam_v2', 'medsam',     'MedSAM'),
        ('ODIR-5K',    'odir5k_baseline_biomedclip','biomedclip', 'BiomedCLIP'),
        ('Kvasir',     'kvasir_baseline',           'resnet50',   'ImageNet ResNet-50'),
        ('Kvasir',     'kvasir_baseline',           'vit',        'ImageNet ViT'),
        ('Kvasir',     'kvasir_baseline_medsam_v2', 'medsam',     'MedSAM'),
        ('Kvasir',     'kvasir_baseline_biomedclip','biomedclip', 'BiomedCLIP'),
    ]
    rows = []
    for ds, prefix, mt, label in configs:
        per_seed = []
        for s in [123, 456, 789, 1024]:
            f = ROOT / 'results' / f'{prefix}_seed{s}' / 'memorization' / f'{mt}_memorization_scores.csv'
            if not f.exists():
                continue
            df = pd.read_csv(f)
            counts = df['class_name'].value_counts()
            df['freq'] = df['class_name'].map(counts / counts.sum())
            rho, _ = spearmanr(df['freq'], df['memorization_score'])
            per_seed.append(rho)
        if not per_seed:
            rows.append({'dataset': ds, 'arch': label, 'rho_mean': np.nan, 'rho_sd': 0.0})
            continue
        rows.append({
            'dataset': ds, 'arch': label,
            'rho_mean': float(np.mean(per_seed)),
            'rho_sd': float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    df = compute_panel_b()
    print(df.round(3))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15],
                          left=0.07, right=0.985, top=0.88, bottom=0.20,
                          wspace=0.30)

    # ---- Panel a: high vs low M(x) MIA --------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(PANEL_A))
    w = 0.36
    high = [p[1] for p in PANEL_A]
    low = [p[2] for p in PANEL_A]
    ax_a.bar(x - w/2, [v*100 for v in high], w, label='High $M(x) > 0.3$',
             color='#c0392b', edgecolor='#7d2017', linewidth=0.6)
    ax_a.bar(x + w/2, [v*100 for v in low], w, label=r'Low $M(x) \leq 0.1$',
             color='#27ae60', edgecolor='#196b3a', linewidth=0.6)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([p[0] for p in PANEL_A], fontsize=9)
    ax_a.set_ylabel('MIA accuracy (%)')
    ax_a.set_title('High-memorization samples are\n31–99 pp more attackable',
                   fontsize=10)
    ax_a.set_ylim(0, 115)
    ax_a.legend(loc='upper right', fontsize=8.5, frameon=False)
    for i, (label, h, l) in enumerate(PANEL_A):
        diff = int(round((h - l) * 100))
        ax_a.text(i, max(h, l) * 100 + 3, f'{diff} pp',
                  ha='center', fontsize=8.5, fontweight='bold')
    fig.text(0.025, 0.93, 'a', fontsize=15, fontweight='bold')

    # ---- Panel b: multi-seed Spearman rho ----------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    datasets = ['HAM10000', 'ChestX-ray', 'ODIR-5K', 'Kvasir']
    archs = ['ImageNet ResNet-50', 'ImageNet ViT', 'MedSAM', 'BiomedCLIP']
    colors = {
        'ImageNet ResNet-50': '#3498db',
        'ImageNet ViT':       '#5dade2',
        'MedSAM':             '#e67e22',
        'BiomedCLIP':         '#a04000',
    }

    n_arch = len(archs)
    bar_w = 0.17
    x = np.arange(len(datasets))

    for i, arch in enumerate(archs):
        offsets = bar_w * (i - (n_arch - 1) / 2)
        means = []
        sds = []
        for ds in datasets:
            row = df[(df['dataset'] == ds) & (df['arch'] == arch)]
            if row.empty or pd.isna(row['rho_mean'].values[0]):
                means.append(np.nan)
                sds.append(0.0)
            else:
                means.append(row['rho_mean'].values[0])
                sds.append(row['rho_sd'].values[0])
        ax_b.bar(x + offsets, means, bar_w, yerr=sds,
                 color=colors[arch], edgecolor='black', linewidth=0.5,
                 label=arch, capsize=2.5,
                 error_kw=dict(elinewidth=0.7, ecolor='black'))

    ax_b.axhline(0, color='black', lw=0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(datasets, fontsize=9.5)
    ax_b.set_ylabel(r'Spearman $\rho$ (frequency vs $M(x)$)')
    ax_b.set_title('Medical-domain pretraining maintains rare-class bias\n'
                   '(more negative = rare classes memorized more)',
                   fontsize=10)
    ax_b.set_ylim(-0.75, 0.10)
    ax_b.legend(loc='lower right', fontsize=7.8, frameon=False, ncol=2)
    fig.text(0.43, 0.93, 'b', fontsize=15, fontweight='bold')

    fig.savefig(OUT_JPG, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_JPG} and {OUT_PDF}')


if __name__ == '__main__':
    main()
