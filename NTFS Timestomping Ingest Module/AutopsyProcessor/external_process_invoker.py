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
    
    def invoke_parsing(self, exported_files_dir, module_output_dir):
        """
        Invoke the Python 3 external processor pipeline for parsing, preprocessing, feature engineering, and model integration.
        
        This method invokes the complete processing pipeline:
        - Stage 1: Raw file parsing ($MFT, $LogFile, $UsnJrnl → CSV files)
        - Stage 2: Data preprocessing (CSV files → grouped_events.csv)
        - Stage 3: Feature engineering (grouped_events.csv → file_features.csv)
        - Stage 4: Model integration (file_features.csv → detection results)
        
        Args:
            exported_files_dir: Path to directory containing exported $MFT, $LogFile, $UsnJrnl
            module_output_dir: Path to module output root directory
                              (contains "Parsed Files", "Grouped Events File", "File Features", "Detection Results" subdirectories)
            
        Returns:
            dict: Result dictionary with keys:
                - 'success': bool - whether the process succeeded
                - 'message': str - status message
                - 'results': dict - processing results if successful, None otherwise
                  Format: {
                    'parsing': {mft, logfile, usnjrnl results},
                    'preprocessing': {grouped_events results},
                    'feature_engineering': {file_features results}
                    'model_integration': {detection results}
                  }
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
            # --output-dir is the MODULE ROOT containing all subdirectories
            args = [
                self.python3_executable,
                timestomp_detector_path,
                '--exported-dir', str(exported_files_dir),
                '--output-dir', str(module_output_dir)
            ]
            
            self.log(Level.INFO, "Invoking Python 3 external processor pipeline")
            self.log(Level.INFO, "Command: " + " ".join(args))
            
            # Spawn subprocess
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
                stderr_lines = stderr.strip().split('\n')
                for line in stderr_lines:
                    if line.strip():
                        self.log(Level.INFO, "External processor: " + line)
            
            # Check return code
            if process.returncode != 0:
                error_msg = "External processor failed with return code {}".format(process.returncode)
                if stderr:
                    error_msg += ": " + stderr[:500]
                    
                self.log(Level.SEVERE, error_msg)
                
                if stdout:
                    self.log(Level.SEVERE, "Stdout: " + stdout[:1000])
                
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
                    self.log(Level.WARNING, "Results missing 'parsing' key")
                    return {
                        'success': False,
                        'message': "Invalid results: missing 'parsing' section",
                        'results': None
                    }
                
                # Check if preprocessing was performed
                has_preprocessing = 'preprocessing' in results and results['preprocessing']
                has_feature_engineering = 'feature_engineering' in results and results['feature_engineering']
                has_model_integration = 'model_integration' in results and results['model_integration']
                
                # Determine overall success
                parsing_results = results.get('parsing', {})
                preprocessing_results = results.get('preprocessing', {})
                feature_engineering_results = results.get('feature_engineering', {})
                model_integration_results = results.get('model_integration', {})
                
                # Log each stage
                parsing_success = any(r.get('success') for r in parsing_results.values() if isinstance(r, dict))
                preprocessing_success = preprocessing_results.get('success', False) if has_preprocessing else None
                feature_success = feature_engineering_results.get('success', False) if has_feature_engineering else None
                model_success = model_integration_results.get('success', False) if has_model_integration else None
                
                self.log(Level.INFO, "Parsing stage completed - results received")
                if has_preprocessing:
                    self.log(Level.INFO, "Preprocessing stage completed - results received")
                if has_feature_engineering:
                    self.log(Level.INFO, "Feature engineering stage completed - results received")
                if has_model_integration:
                    self.log(Level.INFO, "Model integratiion stage completed - results received")
                
                return {
                    'success': True,
                    'message': "Successfully completed processing pipeline (parsing + preprocessing + feature engineering + model integration)",
                    'results': results
                }
                
            except Exception as e:
                error_msg = "Failed to parse process output as JSON: " + str(e)
                self.log(Level.SEVERE, error_msg)
                self.log(Level.SEVERE, "Output: " + stdout[:1000])
                return {
                    'success': False,
                    'message': error_msg,
                    'results': None
                }
        
        except RuntimeError as e:
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
    
    def invoke_preprocessing(self, module_output_dir):
        """
        Invoke just the preprocessing stage (requires parsed CSV files to exist).
        
        This method can be used to re-run preprocessing without re-parsing if the
        parsed CSV files ($MFT_parsed.csv, LogFile_parsed.csv, UsnJrnl_parsed.csv)
        already exist in the "Parsed Files" directory.
        
        Args:
            module_output_dir: Path to module output root directory
        
        Returns:
            dict: Result dictionary with keys:
                - 'success': bool
                - 'message': str
                - 'results': dict with 'preprocessing' section or None
        """
        try:
            if not self.python3_executable or not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Python 3 runtime not available",
                    'results': None
                }
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Path to preprocessing module
            preproc_script = os.path.join(module_dir, "ExternalProcessor", "ModelPackage", "data_preprocessing.py")
            parsed_files_dir = os.path.join(str(module_output_dir), "Parsed Files")
            grouped_events_dir = os.path.join(str(module_output_dir), "Grouped Events File")
            
            if not os.path.exists(preproc_script):
                return {
                    'success': False,
                    'message': "data_preprocessing.py not found",
                    'results': None
                }
            
            args = [
                self.python3_executable,
                preproc_script,
                str(parsed_files_dir),
                str(grouped_events_dir)
            ]
            
            self.log(Level.INFO, "Invoking preprocessing module")
            
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=module_dir
            )
            
            stdout, stderr = process.communicate()
            
            if stderr:
                stderr_lines = stderr.strip().split('\n')
                for line in stderr_lines:
                    if line.strip():
                        self.log(Level.INFO, "Preprocessing: " + line)
            
            if process.returncode != 0:
                error_msg = "Preprocessing failed with code {}".format(process.returncode)
                self.log(Level.SEVERE, error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'results': None
                }
            
            return {
                'success': True,
                'message': "Preprocessing completed successfully",
                'results': {'preprocessing': {'success': True, 'message': stdout}}
            }
        
        except Exception as e:
            error_msg = "Preprocessing error: " + str(e)
            self.log(Level.SEVERE, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }
    
    def invoke_feature_engineering(self, module_output_dir):
        """
        Invoke just the feature engineering stage (requires grouped_events.csv to exist).
        
        This method can be used to re-run feature engineering without re-parsing or 
        re-preprocessing if the grouped_events.csv already exists in the 
        "Grouped Events File" directory.
        
        Args:
            module_output_dir: Path to module output root directory
        
        Returns:
            dict: Result dictionary with keys:
                - 'success': bool
                - 'message': str
                - 'results': dict with 'feature_engineering' section or None
        """
        try:
            if not self.python3_executable or not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Python 3 runtime not available",
                    'results': None
                }
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Path to feature engineering module
            feature_script = os.path.join(module_dir, "ExternalProcessor", "ModelPackage", "feature_engineering.py")
            grouped_events_dir = os.path.join(str(module_output_dir), "Grouped Events File")
            features_dir = os.path.join(str(module_output_dir), "File Features")
            
            if not os.path.exists(feature_script):
                return {
                    'success': False,
                    'message': "feature_engineering.py not found",
                    'results': None
                }
            
            args = [
                self.python3_executable,
                feature_script,
                str(grouped_events_dir),
                str(features_dir)
            ]
            
            self.log(Level.INFO, "Invoking feature engineering module")
            
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=module_dir
            )
            
            stdout, stderr = process.communicate()
            
            if stderr:
                stderr_lines = stderr.strip().split('\n')
                for line in stderr_lines:
                    if line.strip():
                        self.log(Level.INFO, "Feature Engineering: " + line)
            
            if process.returncode != 0:
                error_msg = "Feature engineering failed with code {}".format(process.returncode)
                self.log(Level.SEVERE, error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'results': None
                }
            
            return {
                'success': True,
                'message': "Feature engineering completed successfully",
                'results': {'feature_engineering': {'success': True, 'message': stdout}}
            }
        
        except Exception as e:
            error_msg = "Feature engineering error: " + str(e)
            self.log(Level.SEVERE, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }
    
    def invoke_model_integration(self, module_output_dir):
        """
        Invoke the model integration stage for ML inference.
        
        Note: This is called automatically by invoke_parsing() as STAGE 4.
        This method is provided for flexibility if re-running inference separately.
        
        Args:
            module_output_dir: Path to module output root directory
        
        Returns:
            dict: Result dictionary with keys:
                - 'success': bool
                - 'message': str
                - 'results': dict with 'model_integration' section or None
        """
        try:
            if not self.python3_executable or not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Python 3 runtime not available",
                    'results': None
                }
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Path to model integration module
            model_integration_script = os.path.join(module_dir, "ExternalProcessor", "ModelPackage", "model_integration.py")
            features_dir = os.path.join(str(module_output_dir), "File Features")
            detection_dir = os.path.join(str(module_output_dir), "Detection Results")
            
            if not os.path.exists(model_integration_script):
                return {
                    'success': False,
                    'message': "model_integration.py not found",
                    'results': None
                }
            
            args = [
                self.python3_executable,
                model_integration_script,
                str(features_dir),
                str(detection_dir)
            ]
            
            self.log(Level.INFO, "Invoking model integration module")
            
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=module_dir
            )
            
            stdout, stderr = process.communicate()
            
            if stderr:
                stderr_lines = stderr.strip().split('\n')
                for line in stderr_lines:
                    if line.strip():
                        self.log(Level.INFO, "Model integration: " + line)
            
            if process.returncode != 0:
                error_msg = "Model integration failed with code {}".format(process.returncode)
                self.log(Level.SEVERE, error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'results': None
                }
            
            return {
                'success': True,
                'message': "Model integration completed successfully",
                'results': {'model_integration': {'success': True, 'message': stdout}}
            }
        
        except Exception as e:
            error_msg = "Model integration error: " + str(e)
            self.log(Level.SEVERE, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }
        
        except Exception as e:
            error_msg = "Model integration setup error: " + str(e)
            self.log(Level.SEVERE, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }