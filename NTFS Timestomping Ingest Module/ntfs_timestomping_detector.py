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
        self.export_dir_path = ""
        self.parsed_dir_path = ""
        self.grouped_events_dir_path = ""
        self.file_features_dir_path = ""
        self.detection_results_dir_path = ""
        self.extractor = None
        self.invoker = None

    def startUp(self, context):
        self.context = context
        self.currentCase = Case.getCurrentCase()

        # Create module output directories
        try:
            module_output_dir = os.path.join(self.currentCase.getModuleDirectory(), "NTFS Timestomping Detector")
            self.export_dir_path = os.path.join(module_output_dir, "Exported NTFS Files")
            self.parsed_dir_path = os.path.join(module_output_dir, "Parsed Files")
            self.grouped_events_dir_path = os.path.join(module_output_dir, "Grouped Events File")
            self.file_features_dir_path = os.path.join(module_output_dir, "File Features")
            self.detection_results_dir_path = os.path.join(module_output_dir, "Detection Results") 
            
            for dir_path in [self.export_dir_path, self.parsed_dir_path, self.grouped_events_dir_path, 
                             self.file_features_dir_path, self.detection_results_dir_path]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                
            self.log(Level.INFO, "Module directories created:")
            self.log(Level.INFO, "Export: " + self.export_dir_path)
            self.log(Level.INFO, "Parsed: " + self.parsed_dir_path)
            self.log(Level.INFO, "Grouped Events: " + self.grouped_events_dir_path)
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
        blackboard = Case.getCurrentCase().getSleuthkitCase().getBlackboard()

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
                else:
                    error_msg = "External processor error: {0}".format(invoke_result['message'])
                    self.log(Level.WARNING, error_msg)
                    
            except Exception as e:
                self.log(Level.WARNING, "Error invoking external processor: " + str(e))
        else:
            self.log(Level.WARNING, "External Process Invoker not available - skipping file parsing")

        # --- CREATE ARTIFACTS FOR EXPORTED FILES ---
        progressBar.switchToDeterminate(total_exported)
        artifact_count = 0
        artifact_success = 0
        artifact_failed = 0

        for file_info in exported_files:
            if self.context.isJobCancelled():
                self.log(Level.INFO, "Job cancelled by user during artifact creation")
                return IngestModule.ProcessResult.OK

            artifact_count += 1
            file_obj = file_info['file_object']
            volume_name = file_info['volume_name']
            export_path = file_info['export_path']
            file_type = file_info['file_type']

            self.log(Level.INFO, "Creating artifact " + str(artifact_count) + "/" + str(total_exported) + " for " + file_obj.getName() + " from " + volume_name)

            try:
                # Create artifact attributes
                attrs = Arrays.asList(
                    BlackboardAttribute(
                        BlackboardAttribute.Type.TSK_SET_NAME,
                        TimestompingDetectorDSIngestModuleFactory.moduleName,
                        "NTFS System Files"
                    ),
                    BlackboardAttribute(
                        BlackboardAttribute.Type.TSK_COMMENT,
                        TimestompingDetectorDSIngestModuleFactory.moduleName,
                        "Volume: " + volume_name + " | File Type: " + file_type.upper() + " | Original Path: " + file_obj.getUniquePath() + " | Exported to: " + export_path
                    )
                )

                # Create the artifact
                art = file_obj.newAnalysisResult(
                    BlackboardArtifact.Type.TSK_INTERESTING_FILE_HIT,
                    Score.SCORE_LIKELY_NOTABLE,
                    None,
                    "NTFS System File",
                    None,
                    attrs
                ).getAnalysisResult()

                # Post the artifact to the blackboard
                blackboard.postArtifact(art, TimestompingDetectorDSIngestModuleFactory.moduleName, 
                                       self.context.getJobId())
                
                artifact_success += 1
                self.log(Level.INFO, "Successfully created artifact for " + file_obj.getName())

            except Blackboard.BlackboardException as e:
                artifact_failed += 1
                self.log(Level.SEVERE, "Blackboard error creating artifact for " + file_obj.getName() + ": " + str(e))
            except Exception as e:
                artifact_failed += 1
                self.log(Level.SEVERE, "Error creating artifact for " + file_obj.getName() + ": " + str(e))

            # Update progress bar
            progressBar.progress(artifact_count)

        # --- POST COMPLETION MESSAGE ---
        self.log(Level.INFO, "Artifact creation complete. Success: " + str(artifact_success) + 
                ", Failed: " + str(artifact_failed))

        message_text = ("Successfully processed " + str(extraction_results['complete_volumes']) + " complete volume(s). " +
                       "Exported " + str(total_exported) + " file(s) (" + str(total_failed) + " failed). " +
                       "Created " + str(artifact_success) + " artifact(s). " +
                       "Files exported to: " + self.export_dir_path)
        
        message = IngestMessage.createMessage(IngestMessage.MessageType.DATA, TimestompingDetectorDSIngestModuleFactory.moduleName, message_text)
        IngestServices.getInstance().postMessage(message)

        self.log(Level.INFO, "NTFS Timestomping Detector module completed successfully")
        return IngestModule.ProcessResult.OK