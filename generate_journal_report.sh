#!/bin/bash
# =============================================================================
# JOURNAL REPORT GENERATOR
# =============================================================================
#
# Generates a publication-quality LaTeX report following top-tier medical
# journal standards (Nature Medicine, Lancet Digital Health style).
#
# OUTPUT:
# - journal_report.tex: Complete LaTeX document
# - journal_report.pdf: Compiled PDF
# - figures/: Publication-quality figures (300 DPI)
#
# USAGE:
#   ./generate_journal_report.sh [experiment_name]
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="${1:-ham1000_baseline}"

echo -e "${PURPLE}"
echo "============================================================================="
echo "  JOURNAL REPORT GENERATOR"
echo "  Creating Publication-Quality Report"
echo "============================================================================="
echo -e "${NC}"
echo ""
echo "Experiment: $EXPERIMENT"
echo ""

# Remove old reports
echo -e "${BLUE}[1/4] Cleaning old reports...${NC}"
rm -f summary_report.tex summary_report.pdf summary_report.aux summary_report.log summary_report.out
rm -f summary_report_with_mia.tex summary_report_with_mia.pdf summary_report_with_mia.aux summary_report_with_mia.log summary_report_with_mia.out
rm -f journal_report.aux journal_report.log journal_report.out
echo "  Removed old summary_report* and journal_report auxiliary files"

# Generate report
echo ""
echo -e "${BLUE}[2/4] Generating LaTeX report and figures...${NC}"
python scripts/generate_journal_report.py --experiment "$EXPERIMENT"

# Compile PDF
echo ""
echo -e "${BLUE}[3/4] Compiling PDF (first pass)...${NC}"
if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode journal_report.tex > /dev/null 2>&1 || true

    echo -e "${BLUE}[4/4] Compiling PDF (second pass for references)...${NC}"
    pdflatex -interaction=nonstopmode journal_report.tex > /dev/null 2>&1 || true

    # Check if PDF was created
    if [ -f "journal_report.pdf" ]; then
        echo ""
        echo -e "${GREEN}=============================================================================${NC}"
        echo -e "${GREEN}  REPORT GENERATION COMPLETE${NC}"
        echo -e "${GREEN}=============================================================================${NC}"
        echo ""
        echo "  Output files:"
        echo "    - journal_report.pdf  (Main document)"
        echo "    - journal_report.tex  (LaTeX source)"
        echo "    - figures/            (Publication-quality figures)"
        echo ""
        echo "  Paper structure:"
        echo "    1. Abstract"
        echo "    2. Introduction"
        echo "    3. Methods"
        echo "       - Dataset (HAM10000)"
        echo "       - Model Architectures"
        echo "       - Differential Training Protocol"
        echo "       - Membership Inference Attack"
        echo "       - Rarity Stratification"
        echo "    4. Results"
        echo "       - Model Performance"
        echo "       - Memorization Patterns (Figure 1)"
        echo "       - Rare Disease Privacy Paradox (Figure 2)"
        echo "       - Model Comparison (Figure 3)"
        echo "       - Per-Class Analysis (Figure 4)"
        echo "    5. Discussion"
        echo "       - Clinical Implications"
        echo "       - Pretraining Paradigm Effects"
        echo "       - Recommendations"
        echo "       - Limitations"
        echo "    6. Conclusion"
        echo ""

        # Get page count
        if command -v pdfinfo &> /dev/null; then
            PAGES=$(pdfinfo journal_report.pdf | grep Pages | awk '{print $2}')
            echo "  Document: ${PAGES} pages"
        fi

        echo ""
    else
        echo -e "${RED}  Warning: PDF compilation may have failed. Check journal_report.log${NC}"
    fi
else
    echo -e "${RED}  pdflatex not found. LaTeX report generated but not compiled.${NC}"
    echo "  Install with: sudo apt install texlive-latex-base texlive-latex-extra"
    echo ""
    echo "  To compile manually:"
    echo "    pdflatex journal_report.tex"
    echo "    pdflatex journal_report.tex  # Run twice for references"
fi

# Clean up auxiliary files
rm -f journal_report.aux journal_report.log journal_report.out 2>/dev/null || true

echo ""
