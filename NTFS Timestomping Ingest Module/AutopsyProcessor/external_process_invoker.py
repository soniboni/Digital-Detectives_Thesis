# -- coding: utf-8 --

"""
External Process Invoker - Bridge between Jython (Autopsy) and Python 3

This module acts as a bridge to invoke the Python 3 external processor from the Jython ingest module.
It spawns a subprocess running the Python 3 timestomp_detector.py with the necessary parameters.
"""

import subprocess
import sys
import os
import json

try:
    from java.util.logging import Level
except ImportError:
    Level = None


class ExternalProcessInvoker:
    def __init__(self, logger_obj=None):
        self.logger = logger_obj
        self.python3_executable = self._get_bundled_python_executable()
    
    def log(self, level, msg):
        if self.logger:
            if level is None:
                level = Level.INFO if Level else None
            if level is not None:
                self.logger.log(level, msg)
            else:
                self.logger.info(msg)
        else:
            print(msg)
    
    def _get_bundled_python_executable(self):
        module_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        bundled_python_dir = os.path.join(module_base_dir, "PythonRuntime")
        bundled_python_exe = os.path.join(bundled_python_dir, "python.exe")
        
        if os.path.exists(bundled_python_exe):
            self.log(Level.INFO, "Using bundled Python 3 at: " + bundled_python_exe)
            
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
            error_msg = (
                "Bundled Python 3 runtime not found!\n"
                "Expected location: " + bundled_python_exe + "\n"
                "Please ensure the PythonRuntime folder is present in the module directory.\n"
                "Module base directory: " + module_base_dir
            )
            self.log(Level.SEVERE, error_msg)
            raise RuntimeError(error_msg)
    
    def invoke_parsing(self, exported_files_dir, module_output_dir):
        try:
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
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            timestomp_detector_path = os.path.join(module_dir, "ExternalProcessor", "timestomp_detector.py")
            
            if not os.path.exists(timestomp_detector_path):
                return {
                    'success': False,
                    'message': "timestomp_detector.py not found at: " + timestomp_detector_path,
                    'results': None
                }
            
            args = [
                self.python3_executable,
                timestomp_detector_path,
                '--exported-dir', str(exported_files_dir),
                '--output-dir', str(module_output_dir)
            ]
            
            self.log(Level.INFO, "Invoking Python 3 external processor pipeline")
            self.log(Level.INFO, "Command: " + " ".join(args))
            
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
                        self.log(Level.INFO, "External processor: " + line)
            
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
            
            try:
                if not stdout or not stdout.strip():
                    return {
                        'success': False,
                        'message': "External processor produced no output",
                        'results': None
                    }
                
                results = json.loads(stdout)
                
                try:
                    results = json.loads(stdout)
                except Exception:
                    txt = stdout.strip()
                    last_open = txt.rfind('{')
                    last_close = txt.rfind('}')
                    extracted = None
                    if last_open != -1 and last_close != -1 and last_close > last_open:
                        candidate = txt[last_open:last_close+1]
                        try:
                            results = json.loads(candidate)
                            extracted = candidate
                        except Exception:
                            results = None
                    if results is None:
                        for line in txt.splitlines()[::-1]:
                            s = line.strip()
                            if not s:
                                continue
                            if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                                try:
                                    results = json.loads(s)
                                    extracted = s
                                    break
                                except Exception:
                                    continue

                    if results is None:
                        raise ValueError('No JSON object could be decoded')
                
                if 'parsing' not in results:
                    self.log(Level.WARNING, "Results missing 'parsing' key")
                    return {
                        'success': False,
                        'message': "Invalid results: missing 'parsing' section",
                        'results': None
                    }
                
                has_preprocessing = 'preprocessing' in results and results['preprocessing']
                has_feature_engineering = 'feature_engineering' in results and results['feature_engineering']
                has_model_integration = 'model_integration' in results and results['model_integration']
                
                parsing_results = results.get('parsing', {})
                preprocessing_results = results.get('preprocessing', {})
                feature_engineering_results = results.get('feature_engineering', {})
                model_integration_results = results.get('model_integration', {})
                
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
        try:
            if not self.python3_executable or not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Python 3 runtime not available",
                    'results': None
                }
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
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
        try:
            if not self.python3_executable or not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Python 3 runtime not available",
                    'results': None
                }
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
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
        try:
            if not self.python3_executable or not os.path.exists(self.python3_executable):
                return {
                    'success': False,
                    'message': "Python 3 runtime not available",
                    'results': None
                }
            
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
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