# -*- coding: utf-8 -*-

"""
Artifact Generator for NTFS Timestomping Detector
Generates Autopsy blackboard artifacts from detected_files.csv
"""

import csv
import os
from java.util.logging import Level
from org.sleuthkit.datamodel import BlackboardArtifact, BlackboardAttribute


class ArtifactGenerator(object):
    def __init__(self, case, csv_file_path, logger=None, progress_bar=None, data_source=None):
        self.case = case
        self.csv_file_path = csv_file_path
        self.sleuth_case = case.getSleuthkitCase()
        self.logger = logger
        self.progress_bar = progress_bar
        self.data_source = data_source

    def log(self, level, message):
        if not self.logger:
            print("[" + str(level) + "] " + message)
            return

        try:
            self.logger.logp(level, self.__class__.__name__, '', message)
        except Exception as e:
            print("[ERROR] Logging failed: " + message + " (" + str(e) + ")")

    def create_artifact_type(self):
        try:
            blackboard = self.sleuth_case.getBlackboard()
            artifact_type = blackboard.getOrAddArtifactType(
                "TSK_TIMESTOMPING_DETECTION",
                "NTFS Timestomping Detection"
            )

            if artifact_type:
                self.log(Level.INFO, "Artifact type ready: TSK_TIMESTOMPING_DETECTION")
            else:
                self.log(Level.SEVERE, "Failed to create artifact type")

            return artifact_type

        except Exception as e:
            self.log(Level.SEVERE, "Error creating artifact type: " + str(e))
            return None

    def create_attribute_types(self):
        attribute_types = {}

        attr_definitions = [
            ("TSK_TIMESTOMP_FILEPATH", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.STRING, "File Path"),
            ("TSK_TIMESTOMP_FILENAME", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.STRING, "File Name"),
            ("TSK_TIMESTOMP_CONFIDENCE", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.DOUBLE, "Confidence Score"),
            ("TSK_TIMESTOMP_SEVERITY", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.STRING, "Severity"),
            ("TSK_TIMESTOMP_FORENSIC_SUMMARY", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.STRING, "Forensic Summary"),
            ("TSK_TIMESTOMP_DETECTION_REASONS", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.STRING, "Detection Reasons"),
            ("TSK_TIMESTOMP_RECOMMENDED_ACTION", BlackboardAttribute.TSK_BLACKBOARD_ATTRIBUTE_VALUE_TYPE.STRING, "Recommended Action")
        ]

        try:
            blackboard = self.sleuth_case.getBlackboard()

            for attr_name, attr_value_type, display_name in attr_definitions:
                try:
                    attr_type = blackboard.getOrAddAttributeType(attr_name, attr_value_type, display_name)
                    if attr_type:
                        attribute_types[attr_name] = attr_type
                except Exception as e:
                    self.log(Level.WARNING, "Failed to create attribute type " + attr_name + ": " + str(e))

            self.log(Level.INFO, "Created " + str(len(attribute_types)) + " attribute types")
            return attribute_types

        except Exception as e:
            self.log(Level.SEVERE, "Error creating attribute types: " + str(e))
            return {}

    def process_csv_and_create_artifacts(self):
        if not os.path.exists(self.csv_file_path):
            self.log(Level.SEVERE, "CSV file not found: " + self.csv_file_path)
            return False, 0, 0

        self.log(Level.INFO, "Starting artifact generation from: " + self.csv_file_path)

        artifact_type = self.create_artifact_type()
        if not artifact_type:
            self.log(Level.SEVERE, "Cannot proceed without artifact type")
            return False, 0, 0

        attribute_types = self.create_attribute_types()
        if not attribute_types:
            self.log(Level.SEVERE, "Cannot proceed without attribute types")
            return False, 0, 0

        content_obj = self.data_source

        if not content_obj:
            self.log(Level.SEVERE, "No data source provided for artifact attachment")
            return False, 0, 0

        try:
            self.log(Level.INFO, "Reading CSV file...")
            rows = []

            with open(self.csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    rows.append(row)

            self.log(Level.INFO, "CSV loaded: " + str(len(rows)) + " detection records")

            if not rows:
                self.log(Level.WARNING, "CSV file is empty - no detections to display")
                return True, 0, 0
            
            if self.progress_bar:
                try:
                    self.progress_bar.switchToDeterminate(len(rows))
                except Exception:
                    pass

            artifact_count = 0
            error_count = 0
            module_name = "NTFSTimestompingDetector"
            created_artifacts = []

            for idx, row in enumerate(rows):
                try:
                    artifact = content_obj.newDataArtifact(artifact_type, [])

                    attributes = []

                    file_path = row.get('FilePath', '').strip()
                    if file_path:
                        attributes.append(BlackboardAttribute(
                            attribute_types['TSK_TIMESTOMP_FILEPATH'],
                            module_name,
                            file_path
                        ))

                    file_name = row.get('FileName', '').strip()
                    if file_name:
                        attributes.append(BlackboardAttribute(
                            attribute_types['TSK_TIMESTOMP_FILENAME'],
                            module_name,
                            file_name
                        ))

                    confidence_str = row.get('Confidence', '').strip()
                    if confidence_str:
                        try:
                            confidence_val = float(confidence_str)
                            attributes.append(BlackboardAttribute(
                                attribute_types['TSK_TIMESTOMP_CONFIDENCE'],
                                module_name,
                                confidence_val
                            ))
                        except ValueError:
                            pass

                    severity = row.get('Severity', '').strip()
                    if severity:
                        attributes.append(BlackboardAttribute(
                            attribute_types['TSK_TIMESTOMP_SEVERITY'],
                            module_name,
                            severity
                        ))

                    forensic_summary = row.get('Forensic_Summary', '').strip()
                    if forensic_summary:
                        attributes.append(BlackboardAttribute(
                            attribute_types['TSK_TIMESTOMP_FORENSIC_SUMMARY'],
                            module_name,
                            forensic_summary
                        ))

                    detection_reasons = row.get('Detection_Reasons', '').strip()
                    if detection_reasons:
                        attributes.append(BlackboardAttribute(
                            attribute_types['TSK_TIMESTOMP_DETECTION_REASONS'],
                            module_name,
                            detection_reasons
                        ))

                    recommended_action = row.get('Recommended_Action', '').strip()
                    if recommended_action:
                        attributes.append(BlackboardAttribute(
                            attribute_types['TSK_TIMESTOMP_RECOMMENDED_ACTION'],
                            module_name,
                            recommended_action
                        ))

                    if attributes:
                        artifact.addAttributes(attributes)
                    
                    created_artifacts.append(artifact)
                    artifact_count += 1

                    if self.progress_bar and idx % 50 == 0:
                        try:
                            self.progress_bar.progress(idx)
                        except Exception:
                            pass

                except Exception as e:
                    self.log(Level.WARNING, "Error creating artifact for row " + str(idx + 1) + ": " + str(e))
                    error_count += 1

            try:
                blackboard = self.sleuth_case.getBlackboard()
                if created_artifacts:
                    blackboard.postArtifacts(created_artifacts, module_name)
                    self.log(Level.INFO, "Posted " + str(len(created_artifacts)) + " artifacts to blackboard")
                else:
                    self.log(Level.WARNING, "No artifacts created to post")
            except Exception as e:
                self.log(Level.SEVERE, "Error posting artifacts to blackboard: " + str(e))
                return False, artifact_count, error_count

            self.log(Level.INFO, "Artifact generation complete: " + str(artifact_count) + " created, " + str(error_count) + " errors")

            return True, artifact_count, error_count

        except Exception as e:
            self.log(Level.SEVERE, "Fatal error processing CSV: " + str(e))
            import traceback
            self.log(Level.SEVERE, traceback.format_exc())
            return False, 0, 0