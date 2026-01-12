# -- coding: utf-8 --

"""
External Process Invoker - Bridge between Jython (Autopsy) and Python 3

This module acts as a bridge to invoke the Python 3 external processor from the Jython ingest module.
It spawns a subprocess running the Python 3 timestomp_detector.py with the necessary parameters.

PRODUCTION VERSION: Uses bundled Python 3 runtime for distribution

Environment: Jython 2.7 (Autopsy)
Purpose: Communication bridge between Jython and Python 3 layers
"""

import subprocess
import sys
import os
import json

try:
    from java.util.logging import Level
except ImportError:
    # Fallback for non-Jython environments
    Level = None


class ExternalProcessInvoker:
    """
    Invokes the external Python 3 processor as a subprocess.
    
    Handles communication between Jython ingest module and Python 3 processing layer.
    Uses bundled Python 3 runtime for portability.
    """
    
    def __init__(self, logger_obj=None):
        """
        Initialize the invoker.
        
        Args:
            logger_obj: Optional logger object from Autopsy. If not provided, prints to stdout.
        """
        self.logger = logger_obj
        self.python3_executable = self._get_bundled_python_executable()
    
    def log(self, level, msg):
        """Log message using Autopsy logger or print."""
        if self.logger:
            # Use Level.INFO as default if level is None
            if level is None:
                level = Level.INFO if Level else None
            if level is not None:
                self.logger.log(level, msg)
            else:
                self.logger.info(msg)
        else:
            print(msg)
    
    def _get_bundled_python_executable(self):
        """
        Get the path to the bundled Python 3 executable.
        
        The bundled Python is located in the PythonRuntime folder within the module directory.
        
        Returns:
            str: Path to bundled Python 3 executable
            
        Raises:
            RuntimeError: If bundled Python executable is not found
        """
        # Get the module's base directory
        # __file__ points to AutopsyProcessor/external_process_invoker.py
        # We need to go up two levels to reach the module root
        module_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Path to bundled Python runtime
        bundled_python_dir = os.path.join(module_base_dir, "PythonRuntime")
        bundled_python_exe = os.path.join(bundled_python_dir, "python.exe")
        
        # Verify the bundled Python exists
        if os.path.exists(bundled_python_exe):
            self.log(Level.INFO, "Using bundled Python 3 at: " + bundled_python_exe)
            
            # Verify it's actually Python 3
            try:
                result = subprocess.Popen(
                    [bundled_python_exe, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                stdout, stderr = result.communicate()
                version_output = stdout + stderr
                
                if "Python 3" in version_output:
                    self.log(Level.INFO, "Bundled Python version: " + version_output.strip())
                    return bundled_python_exe
                else:
                    error_msg = "Bundled Python is not Python 3: " + version_output
                    self.log(Level.SEVERE, error_msg)
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = "Failed to verify bundled Python version: " + str(e)
                self.log(Level.SEVERE, error_msg)
                raise RuntimeError(error_msg)
        else:
            # Bundled Python not found - provide helpful error message
            error_msg = (
                "Bundled Python 3 runtime not found!\n"
                "Expected location: " + bundled_python_exe + "\n"
                "Please ensure the PythonRuntime folder is present in the module directory.\n"
                "Module base directory: " + module_base_dir
            )
            self.log(Level.SEVERE, error_msg)
            raise RuntimeError(error_msg)
    
    def invoke_parsing(self, exported_files_dir, parsed_output_dir):
        """
        Invoke the Python 3 timestomp_detector for file parsing.
        
        Args:
            exported_files_dir: Path to directory containing exported $MFT, $LogFile, $UsnJrnl
            parsed_output_dir: Path to directory where parsed CSV files will be saved
            
        Returns:
            dict: Result dictionary with keys:
                - 'success': bool - whether the process succeeded
                - 'message': str - status message
                - 'results': dict - parsed results if successful, None otherwise
                  Format: {'parsing': {'mft': {...}, 'logfile': {...}, 'usnjrnl': {...}}}
        """
        try:
            # Verify Python 3 executable exists
            if not self.python3_executable:
                return {
                    'success': False,
                    'message': "Bundled Python 3 runtime not initialized",
                    'results': None
                }
            
            if not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Bundled Python 3 executable not found at: " + self.python3_executable,
                    'results': None
                }
            
            # Get the path to the external processor
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            timestomp_detector_path = os.path.join(module_dir, "ExternalProcessor", "timestomp_detector.py")
            
            # Verify timestomp_detector.py exists
            if not os.path.exists(timestomp_detector_path):
                return {
                    'success': False,
                    'message': "timestomp_detector.py not found at: " + timestomp_detector_path,
                    'results': None
                }
            
            # Prepare arguments for subprocess
            args = [
                self.python3_executable,
                timestomp_detector_path,
                '--exported-dir', str(exported_files_dir),
                '--output-dir', str(parsed_output_dir)
            ]
            
            self.log(Level.INFO, "Invoking bundled Python 3 external processor")
            self.log(Level.INFO, "Command: " + " ".join(args))
            
            # Spawn subprocess
            # Note: We don't set environment variables as the bundled Python is self-contained
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=module_dir  # Set working directory to module root
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            # Log stderr if present (for debugging)
            if stderr:
                # Only log as warning if there's actual error content
                # (Some libraries write info messages to stderr)
                stderr_lines = stderr.strip().split('\n')
                for line in stderr_lines:
                    if line.strip():
                        self.log(Level.WARNING, "External processor stderr: " + line)
            
            # Check return code
            if process.returncode != 0:
                error_msg = "External processor failed with return code {}".format(process.returncode)
                if stderr:
                    error_msg += ": " + stderr
                else:
                    error_msg += ": No error message available"
                    
                self.log(Level.SEVERE, error_msg)
                
                # Also log stdout in case there's useful info there
                if stdout:
                    self.log(Level.SEVERE, "External processor stdout: " + stdout[:1000])
                
                return {
                    'success': False,
                    'message': error_msg,
                    'results': None
                }
            
            # Parse results from stdout
            try:
                if not stdout or not stdout.strip():
                    return {
                        'success': False,
                        'message': "External processor produced no output",
                        'results': None
                    }
                
                results = json.loads(stdout)
                
                # Validate the results structure
                if 'parsing' not in results:
                    self.log(Level.WARNING, "Results missing 'parsing' key. Output: " + stdout[:500])
                    return {
                        'success': False,
                        'message': "Invalid results format: missing 'parsing' key",
                        'results': None
                    }
                
                return {
                    'success': True,
                    'message': "Successfully completed NTFS file parsing",
                    'results': results
                }
                
            except json.JSONDecodeError as e:
                error_msg = "Failed to parse process output as JSON: " + str(e)
                self.log(Level.SEVERE, error_msg)
                self.log(Level.SEVERE, "Output was: " + stdout[:1000])
                return {
                    'success': False,
                    'message': error_msg,
                    'results': None
                }
        
        except RuntimeError as e:
            # This catches the bundled Python not found error
            error_msg = "Runtime error: " + str(e)
            self.log(Level.SEVERE, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }
        
        except Exception as e:
            error_msg = "Error invoking external processor: " + str(e)
            self.log(Level.SEVERE, error_msg)
            
            # Try to get traceback for debugging
            try:
                import traceback
                tb = traceback.format_exc()
                self.log(Level.SEVERE, "Traceback: " + tb)
            except:
                pass
            
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }