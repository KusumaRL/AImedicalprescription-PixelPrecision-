# ==================== DEPLOYMENT AND DEBUGGING SCRIPT ====================
# Save as debug_and_run.py

import subprocess
import sys
import time
import threading
import webbrowser
import os
from pathlib import Path
import logging
import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        required_packages = ['fastapi', 'uvicorn', 'streamlit', 'requests', 'pandas', 'plotly', 'pydantic']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} is missing")

        if missing_packages:
            logger.info("Installing missing packages...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    *missing_packages
                ])
                logger.info("✅ All dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install dependencies: {e}")
                return False
        return True

    def check_files(self):
        """Check if required files exist"""
        files_to_check = [
            ("main.py", "Backend FastAPI application"),
            ("app.py", "Frontend Streamlit application")
        ]

        missing_files = []
        for filename, description in files_to_check:
            if Path(filename).exists():
                logger.info(f"✅ {filename} found ({description})")
            else:
                missing_files.append((filename, description))
                logger.warning(f"❌ {filename} not found ({description})")

        if missing_files:
            logger.error("Missing required files:")
            for filename, desc in missing_files:
                logger.error(f"  - {filename}: {desc}")
            return False
        return True

    def start_backend(self):
        """Start the FastAPI backend server"""
        logger.info("🔧 Starting AI backend server...")
        try:
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "main:app",
                "--host", "0.0.0.0", "--port", "8000", "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Wait a bit and check if process started successfully
            time.sleep(2)
            if self.backend_process.poll() is None:
                logger.info("✅ Backend server started successfully on port 8000")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                logger.error(f"❌ Backend failed to start: {stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False

    def start_frontend(self):
        """Start the Streamlit frontend"""
        logger.info("🎨 Starting Streamlit frontend...")
        try:
            self.frontend_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", "8501", "--server.headless", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            time.sleep(3)
            if self.frontend_process.poll() is None:
                logger.info("✅ Frontend server started successfully on port 8501")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                logger.error(f"❌ Frontend failed to start: {stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False

    def open_browser(self):
        """Open the browser after a delay"""

        def browser_opener():
            time.sleep(8)  # Give servers time to fully start
            try:
                webbrowser.open("http://localhost:8501")
                logger.info("🌐 Browser opened to http://localhost:8501")
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                logger.info("Please manually open http://localhost:8501 in your browser")

        threading.Thread(target=browser_opener, daemon=True).start()

    def health_check(self):
        """Perform health check on both services"""
        import requests

        # Check backend
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Backend health check passed")
            else:
                logger.warning(f"⚠️ Backend health check failed with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Backend health check failed: {e}")

        # Check frontend (basic connection test)
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Frontend is accessible")
            else:
                logger.info("ℹ️ Frontend is starting up...")
        except requests.exceptions.RequestException:
            logger.info("ℹ️ Frontend is starting up...")

    def cleanup(self):
        """Clean up processes"""
        logger.info("🧹 Cleaning up processes...")

        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                logger.info("✅ Backend process terminated")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                logger.info("🔪 Backend process killed")
            except Exception as e:
                logger.error(f"Error terminating backend: {e}")

        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                logger.info("✅ Frontend process terminated")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                logger.info("🔪 Frontend process killed")
            except Exception as e:
                logger.error(f"Error terminating frontend: {e}")

    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info("🛑 Shutdown signal received")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def run_system(self):
        """Main system runner"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logger.info("🚀 Starting AI Medical Prescription Verification System...")
        logger.info("=" * 60)

        # Step 1: Check dependencies
        logger.info("Step 1: Checking dependencies...")
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed. Exiting.")
            return False

        # Step 2: Check files
        logger.info("Step 2: Checking required files...")
        if not self.check_files():
            logger.error("❌ File check failed. Please ensure main.py and app.py exist.")
            return False

        # Step 3: Start backend
        logger.info("Step 3: Starting backend server...")
        if not self.start_backend():
            logger.error("❌ Backend startup failed. Exiting.")
            return False

        # Step 4: Start frontend
        logger.info("Step 4: Starting frontend server...")
        if not self.start_frontend():
            logger.error("❌ Frontend startup failed.")
            self.cleanup()
            return False

        # Step 5: Open browser
        logger.info("Step 5: Opening browser...")
        self.open_browser()

        # Step 6: Health check
        logger.info("Step 6: Performing health checks...")
        time.sleep(5)
        self.health_check()

        # Keep running and monitor
        logger.info("=" * 60)
        logger.info("🎉 System started successfully!")
        logger.info("🌐 Frontend: http://localhost:8501")
        logger.info("🔧 Backend API: http://localhost:8000")
        logger.info("📖 API Docs: http://localhost:8000/docs")
        logger.info("Press Ctrl+C to stop the system")
        logger.info("=" * 60)

        try:
            while self.running:
                # Monitor processes
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ Backend process died unexpectedly")
                    break

                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.error("❌ Frontend process died unexpectedly")
                    break

                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        finally:
            self.cleanup()

        return True


# ==================== TROUBLESHOOTING GUIDE ====================

def print_troubleshooting_guide():
    """Print comprehensive troubleshooting guide"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    TROUBLESHOOTING GUIDE                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ 🔍 COMMON ISSUES AND SOLUTIONS:                                 ║
║                                                                  ║
║ 1. "ModuleNotFoundError" - Missing Dependencies                  ║
║    Solution: Run 'pip install fastapi uvicorn streamlit         ║
║              requests pandas plotly pydantic'                    ║
║                                                                  ║
║ 2. "Port already in use" - Ports 8000/8501 occupied             ║
║    Solution: Kill processes using those ports or change ports    ║
║    Command: netstat -tulnp | grep :8000                          ║
║             kill -9 <process_id>                                 ║
║                                                                  ║
║ 3. "Connection refused" - Backend not responding                 ║
║    Solution: Check if main.py exists and uvicorn is installed    ║
║    Manual start: uvicorn main:app --reload --port 8000          ║
║                                                                  ║
║ 4. "Streamlit not found" - Streamlit not installed              ║
║    Solution: pip install streamlit                               ║
║    Manual start: streamlit run app.py                           ║
║                                                                  ║
║ 5. API validation errors - Data format issues                   ║
║    Solution: Check input data format and drug names             ║
║    Debug: Check browser developer tools for API responses       ║
║                                                                  ║
║ 6. Slow performance - Large payload or processing               ║
║    Solution: Reduce number of drugs or simplify text            ║
║    Check: API timeout settings (currently 30 seconds)           ║
║                                                                  ║
║ 🔧 DEBUG MODE:                                                  ║
║    Set logging level to DEBUG for more information              ║
║    Add print statements in analysis functions                   ║
║    Check browser console for JavaScript errors                  ║
║                                                                  ║
║ 📋 FILE STRUCTURE CHECK:                                        ║
║    ├── main.py          (FastAPI backend)                       ║
║    ├── app.py           (Streamlit frontend)                    ║
║    ├── debug_and_run.py (This script)                          ║
║    └── requirements.txt (Optional dependencies list)           ║
║                                                                  ║
║ 🆘 EMERGENCY MANUAL START:                                      ║
║    Terminal 1: uvicorn main:app --reload --port 8000           ║
║    Terminal 2: streamlit run app.py --server.port 8501         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Medical Prescription System Manager")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--help-troubleshoot", action="store_true", help="Show troubleshooting guide")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and files")

    args = parser.parse_args()

    if args.help_troubleshoot:
        print_troubleshooting_guide()
        sys.exit(0)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    manager = SystemManager()

    if args.check_only:
        logger.info("🔍 Performing system checks only...")
        deps_ok = manager.check_dependencies()
        files_ok = manager.check_files()

        if deps_ok and files_ok:
            logger.info("✅ All checks passed! System ready to run.")
            sys.exit(0)
        else:
            logger.error("❌ System checks failed. See messages above.")
            sys.exit(1)
    else:
        success = manager.run_system()
        sys.exit(0 if success else 1)


