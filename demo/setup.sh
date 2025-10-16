#!/bin/bash
#
# Setup script for Timestomping Detection Demo
#
# This script creates a virtual environment and installs required packages
#

echo "=========================================="
echo "Timestomping Detection Demo - Setup"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "   Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing virtual environment..."
        rm -rf venv
    else
        echo "   Using existing virtual environment"
        echo ""
        echo "To activate the virtual environment, run:"
        echo "   source venv/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ ! -d "venv" ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install required packages
echo "Installing required packages..."
echo "   - pandas"
echo "   - numpy"
echo "   - scikit-learn"
echo "   - joblib"
echo "   - imbalanced-learn"
echo ""

pip install pandas numpy scikit-learn joblib imbalanced-learn --quiet

if [ $? -eq 0 ]; then
    echo "✓ All packages installed successfully"
else
    echo "❌ Error: Failed to install packages"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the demo scripts:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run Option 1 (Quick Demo):"
echo "   python predict_timestomping.py ../data/processed/Phase\\ 2\\ -\\ Feature\\ Engineering/features_engineered.csv"
echo ""
echo "3. Run Option 2 (Full Pipeline):"
echo "   python full_pipeline_demo.py <logfile.csv> <usnjrnl.csv>"
echo ""
echo "4. When done, deactivate:"
echo "   deactivate"
echo ""