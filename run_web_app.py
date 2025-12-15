#!/usr/bin/env python
"""
Convenience script to run the web application
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required = ['flask', 'numpy', 'pandas', 'arviz']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    return True

def check_data_files():
    """Check if data files exist"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    day_csv = os.path.join(data_dir, 'day.csv')
    hour_csv = os.path.join(data_dir, 'hour.csv')
    
    missing = []
    if not os.path.exists(day_csv):
        missing.append('day.csv')
    if not os.path.exists(hour_csv):
        missing.append('hour.csv')
    
    if missing:
        print(f"Missing data files: {', '.join(missing)}")
        print(f"Expected location: {data_dir}")
        return False
    return True

def main():
    print("=" * 60)
    print("Bayesian Bike Sharing Analysis - Web Application")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies installed")
    
    # Check data files
    print("Checking data files...")
    if not check_data_files():
        print("Warning: Some data files are missing. The app may not work correctly.")
    else:
        print("✓ Data files found")
    
    print()
    print("Starting web server...")
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

