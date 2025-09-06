#!/usr/bin/env python3
"""
Carbon Prompt Playground Launcher
=================================

Simple launcher script for the Carbon Prompt Playground web application.
"""

import os
import sys
import webbrowser
from app import app

def main():
    """Launch the Carbon Prompt Playground."""
    print("🌱" + "="*70)
    print("   CARBON PROMPT PLAYGROUND - Interactive AI Carbon Footprint Explorer")
    print("="*72)
    print()
    print("Starting the web application...")
    print("Loading ML models and data...")
    print("Server will be available at: http://localhost:5001")
    print()
    print("Features:")
    print("   • Real-time carbon emission predictions")
    print("   • Interactive parameter controls")
    print("   • Beautiful D3.js visualizations") 
    print("   • Educational content and tips")
    print("   • Sample prompts and optimization tools")
    print()
    print("How to use:")
    print("   1. Enter your AI prompt in the text area")
    print("   2. Adjust prompt type, complexity, and technical parameters")
    print("   3. Click 'Predict Carbon' to see real-time predictions")
    print("   4. Explore visualizations and optimization suggestions")
    print()
    print("Controls:")
    print("   • Ctrl/Cmd + Enter: Quick predict")
    print("   • ESC: Close modals")
    print("   • Click floating buttons for help and samples")
    print()
    print("Environmental Impact:")
    print("   • Understand carbon footprint of different prompt strategies")
    print("   • Optimize for sustainable AI usage")
    print("   • Learn about energy-carbon relationships")
    print()
    print("="*72)
    
    # Auto-open browser after a short delay
    def open_browser():
        import time
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5001')
            print("Opening browser automatically...")
        except Exception as e:
            print(f"ℹCould not open browser automatically: {e}")
            print("   Please manually navigate to http://localhost:5001")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the Flask application
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5001,
            use_reloader=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nShutting down Carbon Prompt Playground...")
        print("Thank you for exploring sustainable AI!")
    except Exception as e:
        print(f"\nError starting application: {e}")
        print("Please check that all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == '__main__':
    main()