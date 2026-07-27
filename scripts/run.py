#!/usr/bin/env python3
"""
Launch script for MultimodalRAG.

This script provides several ways to start the application:
- Local mode with Streamlit
- Docker mode
- Development mode with hot-reload
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def run_streamlit():
    """Launch the Streamlit application in local mode."""
    cmd = [
        "streamlit", "run", 
        str(ROOT_DIR / "streamlit_app" / "Home.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    print(f"🚀 Starting Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR)


def run_docker_build():
    """Build the Docker image."""
    cmd = ["docker", "build", "-t", "multimodalrag", "."]
    print(f"Docker build: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_docker():
    """Launch the application in Docker mode."""
    # First build the image
    run_docker_build()
    
    # Then run the container
    cmd = [
        "docker", "run", "-d", 
        "-p", "8501:8501", 
        "--name", "multimodalrag_container", 
        "multimodalrag"
    ]
    print(f"Starting Docker: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_docker_compose():
    """Launch the application with Docker Compose."""
    cmd = ["docker-compose", "up", "-d"]
    print(f"Starting Docker Compose: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_dev():
    """Launch in development mode with hot-reload."""
    cmd = [
        "streamlit", "run", 
        str(ROOT_DIR / "streamlit_app" / "Home.py"),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.runOnSave=true",
        "--server.fileWatcherType=auto"
    ]
    print(f"Starting development mode: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch MultimodalRAG")
    parser.add_argument(
        "mode",
        choices=["local", "docker", "docker-compose", "dev", "build"],
        help="Launch mode"
    )
    
    args = parser.parse_args()
    
    print(f"Starting MultimodalRAG in mode: {args.mode}")
    print(f"Working directory: {ROOT_DIR}")
    
    try:
        if args.mode == "local":
            run_streamlit()
        elif args.mode == "docker":
            run_docker()
        elif args.mode == "docker-compose":
            run_docker_compose()
        elif args.mode == "dev":
            run_dev()
        elif args.mode == "build":
            run_docker_build()
    except KeyboardInterrupt:
        print("\nApplication stopped...")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()