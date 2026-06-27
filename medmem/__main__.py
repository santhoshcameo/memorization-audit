"""CLI entry point: python -m medmem or medmem command."""

import argparse
from .pipeline import audit


def main():
    parser = argparse.ArgumentParser(
        prog='medmem',
        description='Medical AI Memorization Audit Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  medmem --data /path/to/images/ --name skin_lesions
  medmem --data /path/to/images/ --csv labels.csv --name chest_xray
  medmem --data /path/to/images/ --name test --quick
  medmem --data /path/to/images/ --name full --models resnet50,vit,mae,medsam
        """)
    parser.add_argument('--data', required=True, help='Path to image directory')
    parser.add_argument('--csv', default=None, help='CSV with image_id and label columns')
    parser.add_argument('--name', required=True, help='Dataset name')
    parser.add_argument('--models', default='resnet50,vit,mae,medsam',
                        help='Comma-separated models (default: all 4)')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--augmentations', type=int, default=5, help='M(x) augmentation passes')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 epochs, resnet50+vit')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    audit(
        data=args.data, name=args.name, csv=args.csv,
        models=models, epochs=args.epochs,
        num_augmentations=args.augmentations,
        seed=args.seed, output=args.output, quick=args.quick,
    )


if __name__ == '__main__':
    main()
