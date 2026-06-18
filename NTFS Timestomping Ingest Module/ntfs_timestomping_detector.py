# -- coding: utf-8 --

"""
Autopsy Python Data Source-level ingest module for detecting NTFS timestomping

This ingest module provides automated detection of NTFS timestomping (timestamp manipulation).
It extracts and parses $MFT, $LogFile, and $UsnJrnl:$J records from a disk image, processing them through a trained machine learning model 
to flag file timestomping activities. The results are displayed in a table within Autopsy and exported as an HTML report and CSV file.
"""

# See https://sleuthkit.org/autopsy/docs/api-docs/latest/index.html for documentation
import jarray
import inspect
import os
import sys

from java.util import Arrays
from java.lang import System
from java.util.logging import Level
from java.io import File

from org.sleuthkit.datamodel import SleuthkitCase
from org.sleuthkit.datamodel import AbstractFile
from org.sleuthkit.datamodel import Score
from org.sleuthkit.datamodel import ReadContentInputStream
from org.sleuthkit.datamodel import BlackboardArtifact
from org.sleuthkit.datamodel import BlackboardAttribute
from org.sleuthkit.autopsy.ingest import IngestModule
from org.sleuthkit.autopsy.ingest.IngestModule import IngestModuleException
from org.sleuthkit.autopsy.ingest import DataSourceIngestModule
from org.sleuthkit.autopsy.ingest import IngestModuleFactoryAdapter
from org.sleuthkit.autopsy.ingest import IngestMessage
from org.sleuthkit.autopsy.ingest import IngestServices
from org.sleuthkit.autopsy.coreutils import Logger
from org.sleuthkit.autopsy.casemodule import Case
from org.sleuthkit.autopsy.casemodule.services import Blackboard

try:
    base_dir = os.path.dirname(__file__)
    if base_dir and base_dir not in sys.path:
        sys.path.insert(0, base_dir)
except Exception:
    pass

# Import Autopsy Processor utilities
from AutopsyProcessor.raw_files_extractor import NTFSFileExtractor
from AutopsyProcessor.external_process_invoker import ExternalProcessInvoker
from AutopsyProcessor.artifact_generator import ArtifactGenerator
from AutopsyProcessor.html_report_generator import HTMLReportGenerator


class TimestompingDetectorDSIngestModuleFactory(IngestModuleFactoryAdapter):

    moduleName = "NTFS Timestomping Detector"

    def getModuleDisplayName(self):
        return self.moduleName

    def getModuleDescription(self):
        return "Detects NTFS timestomping using Machine Learning analysis of $MFT, $LogFile, and $UsnJrnl:$J records."

    def getModuleVersionNumber(self):
        return "1.0"

    def isDataSourceIngestModuleFactory(self):
        return True

    def createDataSourceIngestModule(self, ingestOptions):
        return TimestompingDetectorDSIngestModule()


# Data Source-level ingest module. One gets created per data source.
class TimestompingDetectorDSIngestModule(DataSourceIngestModule):
    _logger = Logger.getLogger(TimestompingDetectorDSIngestModuleFactory.moduleName)

    def log(self, level, msg):
        self._logger.logp(level, self.__class__.__name__, inspect.stack()[1][3], msg)

    def __init__(self):
        self.context = None
        self.extractor = None
        self.invoker = None
        self.export_dir_path = ""
        self.parsed_dir_path = ""
        self.grouped_events_dir_path = ""
        self.file_features_dir_path = ""
        self.detection_results_dir_path = ""

    def startUp(self, context):
        self.context = context
        self.currentCase = Case.getCurrentCase()

        # Create module output directories
        try:
            self.module_output_dir = os.path.join(self.currentCase.getModuleDirectory(), "NTFS Timestomping Detector")
            self.export_dir_path = os.path.join(self.module_output_dir, "Exported Raw NTFS Files")
            self.parsed_dir_path = os.path.join(self.module_output_dir, "Parsed Files")
            self.grouped_events_dir_path = os.path.join(self.module_output_dir, "Grouped Events File")
            self.file_features_dir_path = os.path.join(self.module_output_dir, "File Features")
            self.detection_results_dir_path = os.path.join(self.module_output_dir, "Detection Results") 
            
            for dir_path in [self.export_dir_path, self.parsed_dir_path, self.grouped_events_dir_path, 
                             self.file_features_dir_path, self.detection_results_dir_path]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                
            self.log(Level.INFO, "Module directories created:")
            self.log(Level.INFO, "Exported Raw NTFS Files: " + self.export_dir_path)
            self.log(Level.INFO, "Parsed Files: " + self.parsed_dir_path)
            self.log(Level.INFO, "Grouped Events File: " + self.grouped_events_dir_path)
            self.log(Level.INFO, "File Features: " + self.file_features_dir_path)
            self.log(Level.INFO, "Detection Results: " + self.detection_results_dir_path)
            
        except Exception as e:
            self.log(Level.SEVERE, "Failed to create module output directories: " + str(e))
            raise IngestModuleException("Failed to create module output directories.")
        
        self.extractor = NTFSFileExtractor(self._logger, context)
        self.log(Level.INFO, "NTFS File Extractor initialized")
        
        self.invoker = ExternalProcessInvoker(logger_obj=self._logger)
        self.log(Level.INFO, "External Process Invoker initialized")

    def process(self, dataSource, progressBar):
        """Main processing method"""
        
        progressBar.switchToIndeterminate()
        fileManager = Case.getCurrentCase().getServices().getFileManager()

        self.log(Level.INFO, "Starting NTFS file extraction for data source: " + dataSource.getName())

        # --- EXTRACT ALL FILES USING THE EXTRACTOR ---
        try:
            extraction_results = self.extractor.extract_all_files(fileManager, dataSource, self.export_dir_path)
        except Exception as e:
            self.log(Level.SEVERE, "Fatal error during file extraction: " + str(e))
            return IngestModule.ProcessResult.ERROR

        # --- ANALYZE EXTRACTION RESULTS ---
        analysis = self.extractor.analyze_extraction_results(extraction_results)
        
        if not analysis['should_continue']:
            message = IngestMessage.createMessage(IngestMessage.MessageType.WARNING, TimestompingDetectorDSIngestModuleFactory.moduleName, analysis['message_text'])
            IngestServices.getInstance().postMessage(message)
            return IngestModule.ProcessResult.OK
        
        # Extract data needed for artifact creation
        exported_files = extraction_results['exported_files']
        total_exported = extraction_results['total_exported']
        total_failed = extraction_results['total_failed']

        # --- PARSE NTFS SYSTEM FILES ---
        self.log(Level.INFO, "Starting NTFS system files parsing via external processor")
        if self.invoker is not None:
            try:
                # Get module root directory (parent of subdirectories)
                module_output_dir = os.path.dirname(self.parsed_dir_path)
                
                invoke_result = self.invoker.invoke_parsing(self.export_dir_path, module_output_dir)
                
                if invoke_result['success']:
                    self.log(Level.INFO, "External processor completed successfully")
                    
                    # Log parsing results
                    parsing_results = invoke_result['results'].get('parsing', {})
                    for file_type, result in parsing_results.items():
                        if result['success']:
                            log_msg = "{0}: {1} ({2} records)".format(
                                file_type, 
                                result['message'], 
                                result['records']
                            )
                            self.log(Level.INFO, log_msg)
                        else:
                            error_msg = "{0} error: {1}".format(file_type, result['message'])
                            self.log(Level.WARNING, error_msg)
                    
                    # Log preprocessing results
                    preproc_results = invoke_result['results'].get('preprocessing', {})
                    if preproc_results.get('success'):
                        log_msg = "Preprocessing: {0} ({1} events)".format(
                            preproc_results.get('message', 'Success'),
                            preproc_results.get('event_count', 0)
                        )
                        self.log(Level.INFO, log_msg)
                    else:
                        error_msg = "Data Preprocessing: {0}".format(preproc_results.get('message', 'Unknown error'))
                        self.log(Level.WARNING, error_msg)
                    
                    # Log feature engineering results
                    feature_results = invoke_result['results'].get('feature_engineering', {})
                    if feature_results.get('success'):
                        log_msg = "Feature Engineering: {0} ({1} files analyzed)".format(
                            feature_results.get('message', 'Success'),
                            feature_results.get('file_count', 0)
                        )
                        self.log(Level.INFO, log_msg)
                    else:
                        error_msg = "Feature Engineering: {0}".format(feature_results.get('message', 'Unknown error'))
                        self.log(Level.WARNING, error_msg)
                    
                    # Log model integration results
                    model_results = invoke_result['results'].get('model_integration', {})
                    if model_results.get('success'):
                        log_msg = "Model Integration: {0} ({1} files analyzed, {2} flagged at {3:.2f}%)".format(
                            model_results.get('message', 'Success'),
                            model_results.get('total_files', 0),
                            model_results.get('flagged_files', 0),
                            model_results.get('flag_rate', 0) * 100
                        )
                        self.log(Level.INFO, log_msg)
                        
                        # Log output files
                        output_files = model_results.get('output_files', {})
                        if output_files:
                            self.log(Level.INFO, "Detection output files generated:")
                            if 'detected_files' in output_files:
                                self.log(Level.INFO, "  - Detected Files: " + output_files['detected_files'])
                            if 'files_with_features' in output_files:
                                self.log(Level.INFO, "  - Files with Features: " + output_files['files_with_features'])
                            if 'summary' in output_files:
                                self.log(Level.INFO, "  - Summary Report: " + output_files['summary'])
                    else:
                        warning_msg = "Model Integration: {0}".format(model_results.get('message', 'Unknown error'))
                        self.log(Level.WARNING, warning_msg)
                else:
                    error_msg = "External processor error: {0}".format(invoke_result['message'])
                    self.log(Level.WARNING, error_msg)
                    
            except Exception as e:
                self.log(Level.WARNING, "Error invoking external processor: " + str(e))
        else:
            self.log(Level.WARNING, "External Process Invoker not available - skipping file parsing")

        # --- CREATE ARTIFACTS FOR EXPORTED FILES ---
        # Construct proper path to detected_files.csv from model_integration.py output
        detected_files_csv = os.path.join(self.detection_results_dir_path, "detected_files.csv")
        
        self.log(Level.INFO, "Looking for detected_files.csv at: " + detected_files_csv)
        
        if not os.path.exists(detected_files_csv):
            self.log(Level.WARNING, "detected_files.csv not found at: " + detected_files_csv)
            self.log(Level.WARNING, "Artifact generation skipped - no detections to process")
            
            message_text = ("Processing completed. Exported " + str(total_exported) + " file(s). " +
                           "No detections found or CSV file not available.")
            message = IngestMessage.createMessage(IngestMessage.MessageType.DATA, TimestompingDetectorDSIngestModuleFactory.moduleName, message_text)
            IngestServices.getInstance().postMessage(message)
            self.log(Level.INFO, "NTFS Timestomping Detector module completed")
            return IngestModule.ProcessResult.OK
        
        # Create artifacts from detected files
        self.log(Level.INFO, "Creating blackboard artifacts from detected files")
        case = Case.getCurrentCase()
        generator = ArtifactGenerator(case, detected_files_csv, self._logger, progressBar, dataSource)
        
        success, artifact_created, artifact_errors = generator.process_csv_and_create_artifacts()
        
        if not success:
            self.log(Level.SEVERE, "Artifact generation failed")
            message_text = ("NTFS Timestomping detection encountered errors during artifact creation.")
            message = IngestMessage.createMessage(IngestMessage.MessageType.ERROR, TimestompingDetectorDSIngestModuleFactory.moduleName, message_text)
            IngestServices.getInstance().postMessage(message)
            return IngestModule.ProcessResult.ERROR
        
        # Log success and post completion message
        self.log(Level.INFO, "Artifact creation complete. Success: " + str(artifact_created) + 
                ", Failed: " + str(artifact_errors))

        message_text = ("NTFS Timestomping Detection Complete - " +
                       "Exported " + str(total_exported) + " file(s) (" + str(total_failed) + " failed). " +
                       "Created " + str(artifact_created) + " artifact(s) from detected results. " +
                       "Detection Results are exported to: " + self.detection_results_dir_path)
        
        message = IngestMessage.createMessage(IngestMessage.MessageType.DATA, TimestompingDetectorDSIngestModuleFactory.moduleName, message_text)
        IngestServices.getInstance().postMessage(message)

        # Generate report
        summary_path = os.path.join(self.detection_results_dir_path, "summary.txt")
        report_output_path = os.path.join(self.detection_results_dir_path, "NTFS_Timestomping_Report.html")
        
        if os.path.exists(summary_path):
            try:
                report_generator = HTMLReportGenerator(
                    logger_obj=self._logger,
                    module=self,                    # Passes the ingest module (self)
                    module_name=TimestompingDetectorDSIngestModuleFactory.moduleName
                )

                report_path = report_generator.generate_report_from_summary(
                    summary_path=summary_path,
                    output_path=report_output_path
                )
                self.log(Level.INFO, "HTML report generated: " + report_path)
            except Exception as e:
                self.log(Level.WARNING, "Failed to generate HTML report: " + str(e))
        else:
            self.log(Level.WARNING, "Summary file not found at: " + summary_path)

        self.log(Level.INFO, "NTFS Timestomping Detector Data Source Ingest Module completed successfully")
        return IngestModule.ProcessResult.OK