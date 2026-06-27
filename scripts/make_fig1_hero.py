"""
Regenerate Figure 1 (hero) for the npj Digital Medicine submission.

Layout (clean grid, no overlapping annotations):
  [a] sample images strip            : 4 representative HAM10000 images
  [b] heatmap (5 models x 7 classes) : per-class mean M(x), multi-seed
  [c] grayscale opposite-effects bar : df +753%, akiec -99% (ViT)
  [d] distinctiveness -> MIA bar     : df 0.44->0.89, akiec 0.88->0.27

Inputs (computed from existing 1-pass multi-seed CSVs):
  - HAM10000 per-class M(x), seed-averaged, for ResNet-50, ViT, MAE,
    MedSAM, BiomedCLIP. We use the same per-class average over the
    available seeds for each (model, class) cell.
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('.')
RESULTS = PROJECT_ROOT / 'results'
HAM_RAW = PROJECT_ROOT / 'data' / 'raw' / 'HAM10000'
OUT_JPG = PROJECT_ROOT / 'figures' / 'fig1_hero.jpg'
OUT_PDF = PROJECT_ROOT / 'figures' / 'fig1_hero.pdf'

CLASSES = ['akiec', 'mel', 'bcc', 'bkl', 'df', 'nv', 'vasc']
MODELS = [
    ('ResNet-50',  'ham1000_baseline',           'resnet50',   [123, 456, 789, 1024]),
    ('ViT',        'ham1000_baseline',           'vit',        [123, 456, 789, 1024]),
    ('MAE',        'ham1000_baseline',           'mae',        [123, 456, 789, 1024]),
    ('MedSAM',     'ham1000_baseline_medsam_v2', 'medsam',     [123, 456, 789, 1024]),
    ('BiomedCLIP', 'ham1000_baseline_biomedclip','biomedclip', [123, 456, 789, 1024]),
]

SAMPLE_IMAGES = [
    ('df',    'ISIC_0033256', 'Dermatofibroma\n(1.2%, rarest)'),
    ('mel',   'ISIC_0024700', 'Melanoma\n(11.1%)'),
    ('akiec', 'ISIC_0030825', 'Actinic keratosis\n(3.3%)'),
    ('nv',    'ISIC_0026573', 'Nevus\n(66.9%, common)'),
]

# Authoritative manipulation values (single-seed) used in Sec. 2.4 of paper
PANEL_C = {
    'df':    {'baseline': 0.36, 'gray': 3.07, 'pct': '+753%'},
    'akiec': {'baseline': 1.76, 'gray': 0.03, 'pct': '-99%'},
}
PANEL_D = {
    'df':    {'baseline': 0.44, 'gray': 0.89},
    'akiec': {'baseline': 0.88, 'gray': 0.27},
}


def compute_heatmap() -> pd.DataFrame:
    """Average per-class M(x) across all available seeds (skips missing seeds)."""
    rows = {}
    for label, prefix, mt, seeds in MODELS:
        per_seed = []
        used = []
        for s in seeds:
            f = RESULTS / f'{prefix}_seed{s}' / 'memorization' / f'{mt}_memorization_scores.csv'
            if not f.exists():
                continue
            df = pd.read_csv(f)
            per_seed.append(df.groupby('class_name')['memorization_score'].mean())
            used.append(s)
        if not per_seed:
            raise RuntimeError(f'No CSVs found for {label}')
        avg = pd.concat(per_seed, axis=1).mean(axis=1)
        rows[label] = [avg.get(c, np.nan) for c in CLASSES]
        print(f'  {label}: averaged {len(used)} seeds {used}')
    return pd.DataFrame(rows, index=CLASSES).T


def crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def main():
    heatmap = compute_heatmap()
    print(heatmap.round(2))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size':   9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    fig = plt.figure(figsize=(10.5, 11.0), constrained_layout=False)

    # Manual gridspec with generous spacing to avoid any overlap
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.05, 1.6, 1.4],
        width_ratios=[1.0, 1.0],
        left=0.07, right=0.965, top=0.95, bottom=0.06,
        hspace=0.55, wspace=0.32,
    )

    # ---- Panel a: sample image strip ----------------------------------------
    gs_a = gs[0, :].subgridspec(1, 4, wspace=0.18)
    title_colors = ['#1f77b4', '#000000', '#d62728', '#2ca02c']
    for i, ((cls, image_id, title), c) in enumerate(zip(SAMPLE_IMAGES, title_colors)):
        ax = fig.add_subplot(gs_a[0, i])
        path = HAM_RAW / f'{image_id}.jpg'
        try:
            im = crop_square(Image.open(path))
            ax.imshow(im)
        except Exception:
            ax.text(0.5, 0.5, '(image\nunavailable)', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color('#888888')
            sp.set_linewidth(0.6)
        ax.set_title(title, fontsize=9.5, color=c, pad=4)
    fig.text(0.025, 0.965, 'a', fontsize=15, fontweight='bold')

    # ---- Panel b: heatmap ---------------------------------------------------
    ax_b = fig.add_subplot(gs[1, :])
    data = heatmap.values
    cmap = plt.get_cmap('RdYlGn_r')
    vmax = max(2.7, float(np.nanmax(np.abs(data))))
    vmin = -0.6
    im = ax_b.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            colour = 'white' if (val > 1.6 or val < -0.05) else 'black'
            ax_b.text(j, i, f'{val:+.2f}', ha='center', va='center',
                      fontsize=9.0, color=colour, fontweight='bold')

    ax_b.set_xticks(range(len(CLASSES)))
    ax_b.set_xticklabels(CLASSES, fontsize=9.5)
    ax_b.set_yticks(range(len(MODELS)))
    ax_b.set_yticklabels([m[0] for m in MODELS], fontsize=9.5)
    ax_b.set_title('Per-class memorization $M(x)$ on HAM10000  '
                   '(seed-averaged 1-pass scoring; $n{=}4$ seeds per architecture)',
                   pad=10, fontsize=10.5)

    # Highlight df column without crossing into the colorbar gutter.
    df_col = CLASSES.index('df')
    rect = mpatches.Rectangle(
        (df_col - 0.48, -0.48), 0.96, len(MODELS),
        linewidth=2.2, linestyle='--',
        edgecolor='#1f77b4', facecolor='none', zorder=10,
    )
    ax_b.add_patch(rect)

    # df annotation placed BELOW the heatmap (no overlap with cells/colorbar)
    ax_b.annotate(
        'df: rarest class (1.2%)\nbut not the most memorized',
        xy=(df_col, len(MODELS) - 0.5),
        xytext=(df_col, len(MODELS) + 0.7),
        ha='center', va='top', fontsize=9, color='#1f77b4',
        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
    )

    # Colorbar in its own axes (no overlap with rect)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.025, pad=0.018)
    cbar.set_label('Mean $M(x)$', rotation=270, labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.025, 0.665, 'b', fontsize=15, fontweight='bold')

    # ---- Panel c: opposite effects ------------------------------------------
    ax_c = fig.add_subplot(gs[2, 0])
    classes_c = ['df\n(structure)', 'akiec\n(colour)']
    base_vals = [PANEL_C['df']['baseline'], PANEL_C['akiec']['baseline']]
    gray_vals = [PANEL_C['df']['gray'],     PANEL_C['akiec']['gray']]
    x = np.arange(len(classes_c))
    w = 0.36
    b1 = ax_c.bar(x - w/2, base_vals, w, label='Baseline',
                  color='#9ec5e8', edgecolor='#1f77b4', linewidth=0.8)
    b2 = ax_c.bar(x + w/2, gray_vals, w, label='+ Grayscale',
                  color='#1f77b4', edgecolor='black', linewidth=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(classes_c, fontsize=9.5)
    ax_c.set_ylabel('Mean $M(x)$ (ViT)')
    ax_c.set_title('Same transform, opposite effects', fontsize=10.5)
    ax_c.legend(loc='upper right', fontsize=8.5, frameon=False)
    ax_c.set_ylim(-0.5, 3.6)
    ax_c.axhline(0, color='k', lw=0.5)
    # Annotations for percent change
    ax_c.text(0 + w/2, gray_vals[0] + 0.12, PANEL_C['df']['pct'],
              ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    ax_c.text(1 + w/2, gray_vals[1] + 0.12, PANEL_C['akiec']['pct'],
              ha='center', fontsize=10, color='#d62728', fontweight='bold')
    fig.text(0.025, 0.345, 'c', fontsize=15, fontweight='bold')

    # ---- Panel d: distinctiveness -> MIA ------------------------------------
    ax_d = fig.add_subplot(gs[2, 1])
    base_mia = [PANEL_D['df']['baseline'], PANEL_D['akiec']['baseline']]
    gray_mia = [PANEL_D['df']['gray'],     PANEL_D['akiec']['gray']]
    ax_d.bar(x - w/2, base_mia, w, label='Baseline MIA',
             color='#bcbcbc', edgecolor='#555555', linewidth=0.8)
    ax_d.bar(x + w/2, gray_mia, w, label='+ Grayscale MIA',
             color='#555555', edgecolor='black', linewidth=0.8)
    ax_d.axhline(0.5, color='#888888', lw=0.8, linestyle=':')
    ax_d.text(1.45, 0.515, 'chance', fontsize=8, color='#888888', ha='right')
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(['df', 'akiec'], fontsize=9.5)
    ax_d.set_ylabel('Membership inference AUC')
    ax_d.set_title('Distinctiveness $\\rightarrow$ M$(x)$ $\\rightarrow$ MIA',
                   fontsize=10.5)
    ax_d.set_ylim(0.0, 1.05)
    ax_d.legend(loc='upper right', fontsize=8.5, frameon=False)
    # df: chance -> strong attack annotation
    ax_d.annotate('chance $\\rightarrow$ strong', xy=(0 + w/2, gray_mia[0]),
                  xytext=(0 + w/2, gray_mia[0] + 0.07),
                  ha='center', fontsize=9, color='#1f77b4', fontweight='bold')
    # akiec: below-chance / flipped polarity annotation
    ax_d.annotate('flipped polarity', xy=(1 + w/2, gray_mia[1]),
                  xytext=(1 + w/2, gray_mia[1] + 0.10),
                  ha='center', fontsize=9, color='#d62728', fontweight='bold')
    fig.text(0.51, 0.345, 'd', fontsize=15, fontweight='bold')

    fig.savefig(OUT_JPG, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT_JPG} and {OUT_PDF}')


if __name__ == '__main__':
    main()
