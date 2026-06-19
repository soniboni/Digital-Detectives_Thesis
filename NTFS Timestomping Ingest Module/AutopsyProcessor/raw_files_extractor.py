# -*- coding: utf-8 -*-
"""
NTFS Raw Files Extractor

Extracts NTFS system files ($MFT, $LogFile, $UsnJrnl:$J)
from disk images. It handles volume detection, file searching, validation, and export.
"""

import os
from java.util.logging import Level
from java.io import File
from org.sleuthkit.datamodel import VolumeSystem
from org.sleuthkit.autopsy.casemodule.services import FileManager
from org.sleuthkit.autopsy.datamodel import ContentUtils


class NTFSFileExtractor:

    REQUIRED_FILES = ['$MFT', '$LogFile', '$UsnJrnl:$J']
    
    def __init__(self, logger, context):

        self.logger = logger
        self.context = context
    
    def log(self, level, msg):

        self.logger.logp(level, self.__class__.__name__, "log", msg)
    
    def detect_volumes(self, data_source):

        volumes = []
        
        try:
            children = data_source.getChildren()
            
            for child in children:
                if isinstance(child, VolumeSystem):
                    self.log(Level.INFO, "Found Volume System (Partition Table). Retrieving child volumes...")
                    volumes.extend(child.getChildren())
                else:
                    volumes.append(child)
            
            self.log(Level.INFO, "Total volumes detected: " + str(len(volumes)))
            
        except Exception as e:
            self.log(Level.SEVERE, "Error detecting volumes from the data source: " + str(e))
            raise
        
        return volumes
    
    def is_file_in_volume(self, file_obj, volume_id):

        try:
            current = file_obj
            while current is not None:
                if current.getId() == volume_id:
                    return True
                current = current.getParent()
            return False
        except Exception as e:
            self.log(Level.WARNING, "Error checking file volume: " + str(e))
            return False
    
    def find_file_in_volume(self, file_manager, data_source, file_name, volume_id, volume_name):
        
        try:
            files = file_manager.findFiles(data_source, file_name)
            
            for file_obj in files:
                if self.is_file_in_volume(file_obj, volume_id):
                    self.log(Level.INFO, "Found " + file_name + " in " + volume_name)
                    return file_obj
            
            return None
            
        except Exception as e:
            self.log(Level.WARNING, "Error searching for " + file_name + " in " + volume_name + ": " + str(e))
            return None
    
    def find_usnjrnl_in_volume(self, file_manager, data_source, volume_id, volume_name):
        
        try:
            usnjrnl_j_files = file_manager.findFiles(data_source, "$UsnJrnl:$J")
            
            for file_obj in usnjrnl_j_files:
                if self.is_file_in_volume(file_obj, volume_id):
                    self.log(Level.INFO, "Found $UsnJrnl:$J in " + volume_name)
                    return file_obj
            
            # If not found, search for $UsnJrnl and check for :$J stream
            usnjrnl_files = file_manager.findFiles(data_source, "$UsnJrnl")
            
            for file_obj in usnjrnl_files:
                if self.is_file_in_volume(file_obj, volume_id) and ":$J" in file_obj.getName():
                    self.log(Level.INFO, "Found $UsnJrnl:$J stream in " + volume_name)
                    return file_obj
            
            return None
            
        except Exception as e:
            self.log(Level.WARNING, "Error searching for $UsnJrnl:$J in " + volume_name + ": " + str(e))
            return None
    
    def scan_volume_for_files(self, file_manager, data_source, volume):
        
        volume_id = volume.getId()
        volume_name = volume.getName() if volume.getName() else "Volume_" + str(volume_id)
        
        self.log(Level.INFO, "Scanning volume: " + volume_name + " (ID: " + str(volume_id) + ")")
        
        volume_data = {
            'volume_id': volume_id,
            'volume_name': volume_name,
            'files': {
                'mft': None,
                'logfile': None,
                'usnjrnl': None
            },
            'is_complete': False,
            'missing_files': []
        }
        
        # Search for $MFT
        volume_data['files']['mft'] = self.find_file_in_volume(file_manager, data_source, "$MFT", volume_id, volume_name)
        
        # Search for $LogFile
        volume_data['files']['logfile'] = self.find_file_in_volume(file_manager, data_source, "$LogFile", volume_id, volume_name)
        
        # Search for $UsnJrnl:$J (special case)
        volume_data['files']['usnjrnl'] = self.find_usnjrnl_in_volume(file_manager, data_source, volume_id, volume_name)
        
        # Check if volume has all required files
        if volume_data['files']['mft'] is None:
            volume_data['missing_files'].append('$MFT')
        if volume_data['files']['logfile'] is None:
            volume_data['missing_files'].append('$LogFile')
        if volume_data['files']['usnjrnl'] is None:
            volume_data['missing_files'].append('$UsnJrnl:$J')
        
        volume_data['is_complete'] = len(volume_data['missing_files']) == 0
        
        if not volume_data['is_complete']:
            self.log(Level.WARNING, "Volume " + volume_name + " is incomplete. Missing: " + 
                    ", ".join(volume_data['missing_files']))
        else:
            self.log(Level.INFO, "Volume " + volume_name + " is complete with all required files.")
        
        return volume_data
    
    def generate_unique_filename(self, file_obj, volume_name, volume_id, data_source_id):

        safe_file_name = file_obj.getName().replace(":", "_")
        safe_volume_name = (volume_name.replace(":", "_").replace("/", "_").replace("\\", "_"))
        
        unique_name = (safe_volume_name + "_vid_" + str(volume_id) + "_datasource_" + str(data_source_id) + "_file_" + str(file_obj.getId()) + "_" + safe_file_name)
        
        return unique_name
    
    def export_file(self, file_obj, export_dir, unique_filename):

        try:
            local_path = os.path.join(export_dir, unique_filename)
            local_file = File(local_path)
            
            ContentUtils.writeToFile(file_obj, local_file)
            
            self.log(Level.INFO, "Successfully exported " + file_obj.getName() + " to " + local_path)
            return (True, local_path)
            
        except Exception as e:
            self.log(Level.SEVERE, "Error exporting file " + file_obj.getName() + " to " + local_path + ": " + str(e))
            return (False, None)
    
    def extract_all_files(self, file_manager, data_source, export_dir):
        results = {
            'volumes_scanned': 0,
            'complete_volumes': 0,
            'incomplete_volumes': 0,
            'exported_files': [],
            'total_exported': 0,
            'total_failed': 0,
            'export_dir': export_dir
        }
        
        # STAGE 1: Detect all volumes
        try:
            volumes = self.detect_volumes(data_source)
        except Exception as e:
            self.log(Level.SEVERE, "Failed to detect volumes: " + str(e))
            return results
        
        if len(volumes) == 0:
            self.log(Level.WARNING, "No volumes detected in data source.")
            return results
        
        # STAGE 2: Scan each volume for required files and export
        complete_volume_data = []
        
        for volume in volumes:
            if self.context.isJobCancelled():
                self.log(Level.INFO, "Job cancelled by user.")
                break
            
            try:
                volume_data = self.scan_volume_for_files(file_manager, data_source, volume)
                results['volumes_scanned'] += 1
                
                if volume_data['is_complete']:
                    complete_volume_data.append(volume_data)
                    results['complete_volumes'] += 1
                else:
                    results['incomplete_volumes'] += 1
                    
            except Exception as e:
                self.log(Level.WARNING, "Error scanning volume: " + str(e))
                continue
        
        # STAGE 3: Export files from complete volumes
        for volume_data in complete_volume_data:
            if self.context.isJobCancelled():
                break
            
            volume_name = volume_data['volume_name']
            volume_id = volume_data['volume_id']
            
            # Export each file (MFT, LogFile, UsnJrnl)
            for file_type, file_obj in volume_data['files'].items():
                if file_obj is None:
                    continue
                
                unique_filename = self.generate_unique_filename(
                    file_obj, volume_name, volume_id, data_source.getId()
                )
                
                success, export_path = self.export_file(file_obj, export_dir, unique_filename)
                
                if success:
                    results['total_exported'] += 1
                    results['exported_files'].append({ 
                        'file_object': file_obj,
                        'file_type': file_type,
                        'volume_name': volume_name,
                        'volume_id': volume_id,
                        'export_path': export_path,
                        'unique_filename': unique_filename
                    })
                else:
                    results['total_failed'] += 1
        
        self.log(Level.INFO, "Extraction complete. Exported: " + str(results['total_exported']) + 
                ", Failed: " + str(results['total_failed']))
        
        return results
    
    def analyze_extraction_results(self, extraction_results):
        volumes_scanned = extraction_results['volumes_scanned']
        complete_volumes = extraction_results['complete_volumes']
        incomplete_volumes = extraction_results['incomplete_volumes']
        total_exported = extraction_results['total_exported']
        total_failed = extraction_results['total_failed']
        
        self.log(Level.INFO, "Extraction Summary:")
        self.log(Level.INFO, "  - Volumes Scanned: " + str(volumes_scanned))
        self.log(Level.INFO, "  - Complete Volumes: " + str(complete_volumes))
        self.log(Level.INFO, "  - Incomplete Volumes: " + str(incomplete_volumes))
        self.log(Level.INFO, "  - Files Exported: " + str(total_exported))
        self.log(Level.INFO, "  - Export Failures: " + str(total_failed))
        
        if total_exported == 0:
            if complete_volumes == 0:
                message_text = ("No complete volumes found. " +
                               "Scanned " + str(volumes_scanned) + " volume(s), " + str(incomplete_volumes) + " were incomplete.")
            else:
                message_text = "No files were successfully exported despite finding " + str(complete_volumes) + " complete volume(s). Each volume must contain $MFT, $LogFile, and $UsnJrnl:$J for processing."
            
            self.log(Level.WARNING, "NTFS raw files extraction aborted: " + message_text)
            return {
                'should_continue': False,
                'message_type': 'WARNING',
                'message_text': message_text,
                'log_message': message_text
            }
        
        message_text = ("NTFS raw files extraction successful. Processed " + str(complete_volumes) + " complete volume(s). " +
                       "Exported " + str(total_exported) + " file(s) (" + str(total_failed) + " failed).")
        
        self.log(Level.INFO, message_text)
        return {
            'should_continue': True,
            'message_type': 'INFO',
            'message_text': message_text,
            'log_message': message_text
        }