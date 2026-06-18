# -*- coding: utf-8 -*-
"""
Autopsy ReportModule for NTFS Timestomping Detection HTML Report
Allows opening HTML reports in web browser when clicked in Autopsy's Reports section
"""

from org.sleuthkit.autopsy.report import GeneralReportModule
from org.sleuthkit.autopsy.report import ReportProgressPanel
from org.sleuthkit.autopsy.casemodule import Case
from java.awt import Desktop
from java.net import URI
from java.io import File
import os


class NTFSTimestompingReportModule(GeneralReportModule):
    """
    Autopsy Report Module that opens NTFS Timestomping Detection HTML reports
    in the default web browser when user clicks on the report in Autopsy.
    """
    
    moduleName = "NTFS Timestomping Report Viewer"
    
    def getName(self):
        """Return module name displayed in Autopsy"""
        return self.moduleName
    
    def getDescription(self):
        """Return description shown in Autopsy Reports section"""
        return "Opens NTFS Timestomping Detection HTML reports in default web browser"
    
    def getRelativeFilePath(self):
        """
        Return the relative file path where reports are stored.
        This tells Autopsy where to look for report files.
        """
        return "NTFS Timestomping Detector"
    
    def generateReport(self, baseReportDir):
        """
        Called when user clicks on a report in Autopsy.
        Opens the HTML report in the default web browser.
        
        Args:
            baseReportDir: Base directory path for the report
            
        Returns:
            True if report opened successfully, False otherwise
        """
        try:
            # Construct path to the HTML report
            report_path = os.path.join(baseReportDir, "NTFS_Timestomping_Report.html")
            
            # Verify the report file exists
            report_file = File(report_path)
            if not report_file.exists():
                print("Report file not found: " + report_path)
                return False
            
            # Open the HTML report in default web browser
            if Desktop.isDesktopSupported():
                desktop = Desktop.getDesktop()
                
                # Check if system supports opening files in browser
                if desktop.isSupported(Desktop.Action.BROWSE):
                    # Convert file path to proper file:// URI format
                    # Handle Windows paths with backslashes
                    uri_path = report_path.replace("\\", "/")
                    
                    # On Windows, need to add extra slash for drive letter (e.g., C:/)
                    if uri_path[1:3] == ":/":
                        file_uri = URI("file:///" + uri_path)
                    else:
                        file_uri = URI("file://" + uri_path)
                    
                    # Open in browser
                    desktop.browse(file_uri)
                    print("Opened report in browser: " + report_path)
                    return True
                else:
                    print("System does not support opening URLs in browser")
                    return False
            else:
                print("Desktop not supported on this system")
                return False
                
        except Exception as e:
            print("Error opening report: " + str(e))
            import traceback
            traceback.print_exc()
            return False
    
    def getConfigurationPanel(self):
        """
        Return configuration panel for module settings.
        Not needed for this simple viewer, so return None.
        """
        return None