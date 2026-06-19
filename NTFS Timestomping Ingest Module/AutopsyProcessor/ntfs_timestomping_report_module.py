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
    moduleName = "NTFS Timestomping Report Viewer"
    
    def getName(self):
        return self.moduleName
    
    def getDescription(self):
        return "Opens NTFS Timestomping Detection HTML reports in default web browser"
    
    def getRelativeFilePath(self):
        return "NTFS Timestomping Detector"
    
    def generateReport(self, baseReportDir):
        try:
            report_path = os.path.join(baseReportDir, "NTFS_Timestomping_Report.html")
            
            report_file = File(report_path)
            if not report_file.exists():
                print("Report file not found: " + report_path)
                return False
            
            if Desktop.isDesktopSupported():
                desktop = Desktop.getDesktop()
                
                if desktop.isSupported(Desktop.Action.BROWSE):
                    uri_path = report_path.replace("\\", "/")
                    
                    if uri_path[1:3] == ":/":
                        file_uri = URI("file:///" + uri_path)
                    else:
                        file_uri = URI("file://" + uri_path)
                    
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
        return None