# -- coding: utf-8 --

"""
Autopsy Python Data Source-level ingest module for detecting NTFS timestomping

This ingest module provides automated detection of NTFS timestomping (timestamp manipulation).
It extracts and parses $MFT, $LogFile, and $UsnJrnl:$J records from a disk image, processing them through a trained LightGBM model 
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

    def process(self, dataSource, progressBar):
        """Main processing method"""

        progressBar.switchToIndeterminate()

        # All entries are summarised in the single post-completion IngestMessage at the end.
        errors = []

        # Tracks whether artifact creation succeeded so the final message can reflect it.
        artifact_created = 0
        artifact_errors = 0
        final_result = IngestModule.ProcessResult.OK

        # STAGE 1: Obtain file manager
        try:
            fileManager = Case.getCurrentCase().getServices().getFileManager()
        except Exception as e:
            err_detail = "Could not obtain FileManager: " + str(e)
            self.log(Level.SEVERE, err_detail)
            errors.append(("FileManager Initialization", err_detail))
           
            self._post_completion_message(errors, artifact_created, artifact_errors,
                                          fatal=True, final_result=IngestModule.ProcessResult.ERROR)
            return IngestModule.ProcessResult.ERROR

        self.log(Level.INFO, "Starting NTFS file extraction for data source: " + dataSource.getName())

        # STAGE 2: Extract NTFS raw files
        extraction_results = None
        try:
            extraction_results = self.extractor.extract_all_files(fileManager, dataSource, self.export_dir_path)
        except Exception as e:
            err_detail = "Fatal error during file extraction: " + str(e)
            self.log(Level.SEVERE, err_detail)
            errors.append(("NTFS File Extraction", err_detail))
            self._post_completion_message(errors, artifact_created, artifact_errors,
                                          fatal=True, final_result=IngestModule.ProcessResult.ERROR)
            return IngestModule.ProcessResult.ERROR

        # STAGE 3: Analyze extraction results
        try:
            analysis = self.extractor.analyze_extraction_results(extraction_results)
        except Exception as e:
            err_detail = "Error analysing extraction results: " + str(e)
            self.log(Level.SEVERE, err_detail)
            errors.append(("Extraction Analysis", err_detail))
            self._post_completion_message(errors, artifact_created, artifact_errors,
                                          fatal=True, final_result=IngestModule.ProcessResult.ERROR)
            return IngestModule.ProcessResult.ERROR

        if not analysis['should_continue']:
            subject = "NTFS Timestomping Detection skipped - no eligible files found."
            message = IngestMessage.createMessage(
                IngestMessage.MessageType.WARNING,
                TimestompingDetectorDSIngestModuleFactory.moduleName,
                subject,
                analysis['message_text']
            )
            IngestServices.getInstance().postMessage(message)
            return IngestModule.ProcessResult.OK

        # STAGE 4: External Python 3 processes (parsing → preprocessing → feature engineering → model integration)
        self.log(Level.INFO, "Starting NTFS system files parsing via external processor")

        if self.invoker is None:
            err_detail = "ExternalProcessInvoker is not initialised — file parsing skipped."
            self.log(Level.WARNING, err_detail)
            errors.append(("External Process Invoker", err_detail))
        else:
            try:
                module_output_dir = os.path.dirname(self.parsed_dir_path)
                invoke_result = self.invoker.invoke_parsing(self.export_dir_path, module_output_dir)

                if invoke_result['success']:
                    self.log(Level.INFO, "External processor completed successfully")

                    # STAGE 4.1: Per-file-type parsing results
                    parsing_results = invoke_result['results'].get('parsing', {})
                    for file_type, result in parsing_results.items():
                        if result['success']:
                            self.log(Level.INFO, "{0}: {1} ({2} records)".format(
                                file_type, result['message'], result['records']))
                        else:
                            err_detail = "{0} parsing failed: {1}".format(file_type, result['message'])
                            self.log(Level.WARNING, err_detail)
                            errors.append(("Parsing - " + file_type, err_detail))

                    # STAGE 4.2: Preprocessing
                    preproc_results = invoke_result['results'].get('preprocessing', {})
                    if preproc_results.get('success'):
                        self.log(Level.INFO, "Preprocessing: {0} ({1} events)".format(
                            preproc_results.get('message', 'Success'),
                            preproc_results.get('event_count', 0)))
                    else:
                        err_detail = "Preprocessing failed: {0}".format(
                            preproc_results.get('message', 'Unknown error'))
                        self.log(Level.WARNING, err_detail)
                        errors.append(("Data Preprocessing", err_detail))

                    # STAGE 4.3: Feature engineering
                    feature_results = invoke_result['results'].get('feature_engineering', {})
                    if feature_results.get('success'):
                        self.log(Level.INFO, "Feature Engineering: {0} ({1} files analyzed)".format(
                            feature_results.get('message', 'Success'),
                            feature_results.get('file_count', 0)))
                    else:
                        err_detail = "Feature Engineering failed: {0}".format(
                            feature_results.get('message', 'Unknown error'))
                        self.log(Level.WARNING, err_detail)
                        errors.append(("Feature Engineering", err_detail))

                    # STAGE 4.4: Model integration
                    model_results = invoke_result['results'].get('model_integration', {})
                    if model_results.get('success'):
                        self.log(Level.INFO,
                            "Model Integration: {0} ({1} files analyzed, {2} flagged at {3:.2f}%)".format(
                                model_results.get('message', 'Success'),
                                model_results.get('total_files', 0),
                                model_results.get('flagged_files', 0),
                                model_results.get('flag_rate', 0) * 100))
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
                        err_detail = "Model Integration failed: {0}".format(
                            model_results.get('message', 'Unknown error'))
                        self.log(Level.WARNING, err_detail)
                        errors.append(("Model Integration", err_detail))

                else:
                    err_detail = "External processor returned an error: {0}".format(invoke_result['message'])
                    self.log(Level.WARNING, err_detail)
                    errors.append(("External Processor", err_detail))

            except Exception as e:
                err_detail = "Unexpected exception while invoking external processor: " + str(e)
                self.log(Level.WARNING, err_detail)
                errors.append(("External Processor (exception)", err_detail))

        # STAGE 5: Locate detected_files.csv
        detected_files_csv = os.path.join(self.detection_results_dir_path, "detected_files.csv")
        self.log(Level.INFO, "Looking for detected_files.csv at: " + detected_files_csv)

        if not os.path.exists(detected_files_csv):
            err_detail = ("detected_files.csv not found at: " + detected_files_csv +
                          " — artifact generation skipped.")
            self.log(Level.WARNING, err_detail)
            errors.append(("Artifact Generation", err_detail))
            self._post_completion_message(errors, artifact_created, artifact_errors, fatal=False, final_result=IngestModule.ProcessResult.OK)
            return IngestModule.ProcessResult.OK

        # STAGE 6: Create blackboard artifacts
        self.log(Level.INFO, "Creating blackboard artifacts from detected files")
        try:
            case = Case.getCurrentCase()
            generator = ArtifactGenerator(case, detected_files_csv, self._logger, progressBar, dataSource)
            success, artifact_created, artifact_errors = generator.process_csv_and_create_artifacts()
        except Exception as e:
            err_detail = "Unexpected exception during artifact generation: " + str(e)
            self.log(Level.SEVERE, err_detail)
            errors.append(("Artifact Generation (exception)", err_detail))
            self._post_completion_message(errors, artifact_created, artifact_errors, fatal=True, final_result=IngestModule.ProcessResult.ERROR)
            return IngestModule.ProcessResult.ERROR

        if not success:
            err_detail = ("Artifact generation returned a failure status — " +
                          str(artifact_errors) + " artifact(s) could not be created.")
            self.log(Level.SEVERE, err_detail)
            errors.append(("Artifact Generation", err_detail))
            final_result = IngestModule.ProcessResult.ERROR
        else:
            self.log(Level.INFO, "Artifact creation complete. Success: " + str(artifact_created) +
                     ", Failed: " + str(artifact_errors))
            if artifact_errors > 0:
                err_detail = (str(artifact_errors) + " artifact(s) failed to be created during processing.")
                self.log(Level.WARNING, err_detail)
                errors.append(("Artifact Generation (partial)", err_detail))

        # STAGE 7: Generate HTML report
        summary_path = os.path.join(self.detection_results_dir_path, "summary.txt")
        report_output_path = os.path.join(self.detection_results_dir_path, "NTFS_Timestomping_Report.html")

        if os.path.exists(summary_path):
            try:
                report_generator = HTMLReportGenerator(
                    logger_obj=self._logger,
                    module=self,
                    module_name=TimestompingDetectorDSIngestModuleFactory.moduleName
                )
                report_path = report_generator.generate_report_from_summary(
                    summary_path=summary_path,
                    output_path=report_output_path
                )
                self.log(Level.INFO, "HTML report generated: " + report_path)
            except Exception as e:
                err_detail = "Failed to generate HTML report: " + str(e)
                self.log(Level.WARNING, err_detail)
                errors.append(("HTML Report Generation", err_detail))
        else:
            err_detail = "Summary file not found at: " + summary_path + " — HTML report skipped."
            self.log(Level.WARNING, err_detail)
            errors.append(("HTML Report Generation", err_detail))

        # Post-completion message
        self._post_completion_message(errors, artifact_created, artifact_errors, fatal=(final_result == IngestModule.ProcessResult.ERROR), final_result=final_result)

        if final_result == IngestModule.ProcessResult.OK:
            self.log(Level.INFO, "NTFS Timestomping Detector Data Source Ingest Module completed successfully")
        else:
            self.log(Level.WARNING, "NTFS Timestomping Detector Data Source Ingest Module completed with errors")

        return final_result

    # Build and post the single end-of-run IngestMessage
    def _post_completion_message(self, errors, artifact_created, artifact_errors, fatal, final_result):
        MODULE = TimestompingDetectorDSIngestModuleFactory.moduleName

        # Subject line (shown in the inbox)
        if fatal and artifact_created == 0 and artifact_errors == 0:
            subject = "NTFS Timestomping Detection FAILED - module aborted early."
        elif errors:
            subject = ("NTFS Timestomping Detection completed with {0} error(s) - "
                       "{1} artifact(s) created, {2} failed.".format(
                           len(errors), artifact_created, artifact_errors))
        else:
            subject = ("NTFS Timestomping Detection Complete - "
                       "{0} artifact(s) created, {1} failed.".format(artifact_created, artifact_errors))

        # Details body (shown when the message is opened)
        detail_lines = [subject]

        if artifact_created > 0 or artifact_errors == 0:
            detail_lines.append("Detection Results exported to: " + self.detection_results_dir_path)

        if errors:
            detail_lines.append("")
            detail_lines.append("--- Error Summary ({0} total) ---".format(len(errors)))
            for idx, (stage, detail) in enumerate(errors, start=1):
                detail_lines.append("[{0}] {1}: {2}".format(idx, stage, detail))

        details_text = "\n".join(detail_lines)

        # Message type
        if fatal or final_result == IngestModule.ProcessResult.ERROR:
            msg_type = IngestMessage.MessageType.ERROR
        elif errors:
            msg_type = IngestMessage.MessageType.WARNING
        else:
            msg_type = IngestMessage.MessageType.DATA

        message = IngestMessage.createMessage(msg_type, TimestompingDetectorDSIngestModuleFactory.moduleName, subject, details_text)
        IngestServices.getInstance().postMessage(message)