#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "====================================="
echo "Building React Frontend Assets"
echo "====================================="
cd frontend
npm install
npm run build
cd ..

echo "====================================="
echo "Installing Python Backend Packages"
echo "====================================="
pip install -r backend/requirements.txt

echo "====================================="
echo "Build Process Completed Successfully!"
echo "====================================="
