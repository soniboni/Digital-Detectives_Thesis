# -*- coding: utf-8 -*-
"""
HTML Report Generator for NTFS Timestomping Detection

Creates comprehensive HTML summary reports with statistics and visualizations.
No external dependencies, all charting is done with inline JavaScript.
"""

import os
import re
from datetime import datetime
from java.util.logging import Level


class HTMLReportGenerator:
    def __init__(self, logger_obj=None, module=None, module_name=None):
        self.logger = logger_obj
        self.module = module
        self.module_name = module_name
    
    def log(self, level, msg):
        if self.logger:
            self.logger.logp(level, self.__class__.__name__, "log", msg)
        else:
            print("[{0}] {1}".format(level, msg))
    
    def parse_summary_file(self, summary_path):
        try:
            self.log(Level.INFO, "Parsing summary file: " + summary_path)
            
            with open(summary_path, 'r') as f:
                content = f.read()
            
            data = {
                'analysis_date': '',
                'model_used': '',
                'threshold': '',
                'total_files': 0,
                'files_flagged': 0,
                'flag_percentage': 0.0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0,
                'crr_percentage': 0.0,
                'nni': 0.0,
                'all_flagged_files': [],
                'recommended_next_steps': [],
                'indicators': {},
                'methodology_features': 0,
                'output_files': []
            }
            
            match = re.search(r'Analysis Date:\s+(.+)', content)
            if match:
                data['analysis_date'] = match.group(1).strip()
            
            match = re.search(r'Model Used:\s+(.+)', content)
            if match:
                data['model_used'] = match.group(1).strip()
            
            match = re.search(r'Threshold:\s+(.+)', content)
            if match:
                data['threshold'] = match.group(1).strip()
            
            match = re.search(r'Total Files Analyzed:\s+([\d,]+)', content)
            if match:
                data['total_files'] = int(match.group(1).replace(',', ''))
            
            match = re.search(r'Files Flagged:\s+([\d,]+)\s+\(([\d.]+)%\)', content)
            if match:
                data['files_flagged'] = int(match.group(1).replace(',', ''))
                data['flag_percentage'] = float(match.group(2))
            
            match = re.search(r'High Confidence \(>0\.5\):\s+(\d+)', content)
            if match:
                data['high_confidence'] = int(match.group(1))
            
            match = re.search(r'Medium Confidence \(0\.1-0\.5\):\s+(\d+)', content)
            if match:
                data['medium_confidence'] = int(match.group(1))
            
            match = re.search(r'Low Confidence \([^)]+\):\s+(\d+)', content)
            if match:
                data['low_confidence'] = int(match.group(1))
            
            match = re.search(r'HIGH Severity:\s+(\d+)', content)
            if match:
                data['high_severity'] = int(match.group(1))
            
            match = re.search(r'MEDIUM Severity:\s+(\d+)', content)
            if match:
                data['medium_severity'] = int(match.group(1))
            
            match = re.search(r'LOW Severity:\s+(\d+)', content)
            if match:
                data['low_severity'] = int(match.group(1))
            
            match = re.search(r'Candidate Reduction Rate:\s+([\d.]+)%', content)
            if match:
                data['crr_percentage'] = float(match.group(1))
            
            match = re.search(r'Number Needed to Investigate:\s+~([\d.]+)', content)
            if match:
                data['nni'] = float(match.group(1))
            
            files_start = content.find('ALL FLAGGED FILES')
            if files_start == -1:
                files_start = content.find('ALL FLAGGED FILES')
            detection_summary_start = content.find('DETECTION INDICATOR SUMMARY')

            if files_start != -1 and detection_summary_start != -1:
                files_section = content[files_start:detection_summary_start]
                lines = files_section.split('\n')
                in_data = False

                for line in lines:
                    line = line.strip()
                    if line.startswith('Rank'):
                        in_data = True
                        continue
                    if line.startswith('-'):
                        continue
                    if not line or in_data is False:
                        continue
                    if line.startswith('---') or line.startswith('==='):
                        break

                    parts = line.split(None, 4)
                    if len(parts) >= 5:
                        try:
                            data['all_flagged_files'].append({
                                'rank': int(parts[0]),
                                'confidence': float(parts[1]),
                                'severity': parts[2],
                                'indicators': int(parts[3]),
                                'filename': parts[4]
                            })
                        except (ValueError, IndexError):
                            self.log(Level.WARNING, "Failed to parse flagged-file line: " + line)
                            continue

            indicators_start = content.find('DETECTION INDICATOR SUMMARY')
            explanations_start = content.find('INDICATOR EXPLANATIONS')
            
            if indicators_start != -1 and explanations_start != -1:
                indicators_section = content[indicators_start:explanations_start]
                lines = indicators_section.split('\n')
                in_data = False

                for line in lines:
                    if 'Indicator Type' in line and 'Count' in line:
                        in_data = True
                        continue
                    if line.strip().startswith('-'):
                        continue
                    if not line.strip() or in_data is False:
                        continue
                    if line.strip().startswith('---') or line.strip().startswith('==='):
                        break
                    
                    match = re.match(r'(.+?)\s{2,}(\d+)\s{2,}(.+)', line)
                    if match:
                        indicator_name = match.group(1).strip()
                        indicator_count = int(match.group(2))
                        data['indicators'][indicator_name] = indicator_count
                        
                        self.log(Level.INFO, "Found indicator: {0} = {1}".format(indicator_name, indicator_count))

            match = re.search(r'Features analyzed:\s+(\d+)', content)
            if match:
                data['methodology_features'] = int(match.group(1))

            rec_start = content.find('RECOMMENDED NEXT STEPS')
            methodology_start = content.find('METHODOLOGY')
            if rec_start != -1 and methodology_start != -1:
                rec_section = content[rec_start:methodology_start]
                for line in rec_section.split('\n'):
                    m = re.match(r'\s*\d+\.\s+(.+)', line)
                    if m:
                        data['recommended_next_steps'].append(m.group(1).strip())

            output_start = content.find('OUTPUT FILES GENERATED')
            if output_start != -1:
                output_section = content[output_start:]
                gen_idx = output_section.find('====', 25)
                if gen_idx != -1:
                    output_section = output_section[:gen_idx]
                    
                # Directory containing the HTML report and generated files
                report_dir = os.path.dirname(summary_path)    
                    
                entries = re.split(r'\n(?=\s*\d+\.\s)', output_section)
                for entry in entries:
                    m = re.match(r'\s*\d+\.\s+(\S+?)\s+-\s+(.+)', entry.strip(), re.DOTALL)
                    if m:
                        name = m.group(1).strip()
                        if name == 'autopsy.log.0':
                            continue
                        desc = re.sub(r'\s+', ' ', m.group(2)).strip()
                        file_path = os.path.join(report_dir, name)
                        
                        data['output_files'].append({'name': name, 'path': file_path, 'description': desc})

            if not any(f['name'] == 'autopsy.log.0' for f in data['output_files']):
                try:
                    report_dir = os.path.dirname(summary_path)  # Detection Results folder
                    module_out_dir = os.path.dirname(report_dir)  # NTFS Timestomping Detector folder
                    module_parent_dir = os.path.dirname(module_out_dir)  # ModuleOutput folder
                    case_root_dir = os.path.dirname(module_parent_dir)  # Case root directory
                    
                    autopsy_log_path = os.path.join(
                        case_root_dir,
                        "Log",
                        "autopsy.log.0"
                    )

                    data['output_files'].append({
                        'name': 'autopsy.log.0',
                        'path': autopsy_log_path,
                        'description': "Autopsy's main runtime log file, which records runtime "
                                       "messages, errors, warnings, and status updates generated "
                                       "by Autopsy while working on a case."
                    })
                except Exception as e:
                    self.log(Level.WARNING, "Failed to add autopsy.log.0: " + str(e))
            
            for f in data['output_files']:
                self.log(
                    Level.INFO,
                    "Output File Link: {0} -> {1}".format(
                        f['name'],
                        f['path']
                    )
                )
            
            self.log(Level.INFO, "Successfully parsed summary file")
            self.log(Level.INFO, "Total files: {0}, Flagged: {1}".format(data['total_files'], data['files_flagged']))
            self.log(Level.INFO, "Indicators found: {0}".format(len(data['indicators'])))
            self.log(Level.INFO, "Flagged files: {0}".format(len(data['all_flagged_files'])))

            return data
            
        except Exception as e:
            self.log(Level.SEVERE, "Error parsing summary file: " + str(e))
            raise
    
    def generate_html_report(self, data, output_path):
        try:
            self.log(Level.INFO, "Generating HTML report at: " + output_path)
            
            def js_array_str(values):
                return '[' + ','.join(values) + ']'
            
            def esc(x):
                try:
                    return str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
                except Exception:
                    return ''
            
            indicator_labels = []
            indicator_values = []
            for indicator_name, count in sorted(data['indicators'].items(), key=lambda x: x[1], reverse=True):
                short_name = indicator_name
                if len(short_name) > 35:
                    short_name = short_name[:32] + "..."

                indicator_labels.append(short_name)
                indicator_values.append(count)
            
            files_cleared = data['total_files'] - data['files_flagged']
            
            html = []
            html.append('<!DOCTYPE html>')
            html.append('<html lang="en">')
            html.append('<head>')
            html.append('<meta charset="utf-8"/>')
            html.append('<meta name="viewport" content="width=device-width, initial-scale=1"/>')
            html.append('<title>NTFS Timestomping Detection Report</title>')
            
            # CSS Styles
            html.append('<style>')
            html.append('*{margin:0;padding:0;box-sizing:border-box}')
            html.append('body{font-family:Arial,Helvetica,sans-serif;background:#ffffff;color:#000000;line-height:1.7;padding:25px}')
            html.append('.container{max-width:1400px;margin:0 auto;background:#ffffff}')
            
            # Header
            html.append('.header{background:#2e5c8a;color:#ffffff;padding:30px 35px;margin-bottom:30px;border-bottom:3px solid #1a3a5a}')
            html.append('.header h1{font-size:32px;font-weight:bold;margin-bottom:10px;letter-spacing:0.3px}')
            html.append('.header .subtitle{font-size:16px;opacity:0.95;font-weight:normal;margin-bottom:20px}')
            html.append('.header .disclaimer{font-size:14px;color:#ffffff;opacity:0.95;font-weight:normal;font-style:italic}')

            # Metadata bar
            html.append('.metadata{background:#f0f0f0;border:1px solid #cccccc;padding:20px 25px;margin-bottom:25px}')
            html.append('.metadata-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px 30px}')
            html.append('.metadata-item{display:flex;flex-direction:column}')
            html.append('.metadata-label{font-size:12px;color:#555555;font-weight:bold;text-transform:uppercase;margin-bottom:5px;letter-spacing:0.5px}')
            html.append('.metadata-value{font-size:15px;color:#000000;font-weight:500}')

            # Stats cards
            html.append('.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px;margin-bottom:30px}')
            html.append('.stat-card{background:#ffffff;border:1px solid #cccccc;padding:25px;transition:box-shadow 0.2s,transform 0.1s}')
            html.append('.stat-card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.15);transform:translateY(-2px)}')
            html.append('.stat-card h3{font-size:13px;color:#2e5c8a;font-weight:bold;text-transform:uppercase;margin-bottom:12px;letter-spacing:0.5px}')
            html.append('.stat-card .value{font-size:38px;font-weight:bold;color:#000000;line-height:1.1;margin-bottom:8px}')
            html.append('.stat-card .subtext{font-size:14px;color:#666666}')
            
            # Generated Output Files section
            html.append('.output-files{background:#f8f8f8;border:1px solid #dddddd;padding:22px;margin-top:25px;margin-bottom:25px}')
            html.append('.output-files h2{font-size:20px;color:#000000;font-weight:bold;margin-bottom:15px}')
            html.append('.output-files ul{list-style:none;padding:0}')
            html.append('.output-files li{padding:12px 15px;margin-bottom:8px;background:#ffffff;border:1px solid #dddddd;font-size:14px}')
            html.append('.output-files li strong{color:#2e5c8a;font-weight:bold}')

            # Charts
            html.append('.charts-grid{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:25px}')
            html.append('.chart-card{background:#ffffff;border:1px solid #cccccc;padding:20px;transition:box-shadow 0.2s}')
            html.append('.chart-card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.15)}')
            html.append('.chart-card h2{font-size:16px;color:#000000;font-weight:bold;margin-bottom:15px;padding-bottom:8px;border-bottom:2px solid #2e5c8a}')
            html.append('.chart-card canvas{max-width:100%;height:auto}')
            html.append('.chart-card .chart-note{font-size:14px;color:#666666;margin-top:10px;}')
            html.append('.full-width{grid-column:1/-1}')
            
            # Section styling
            html.append('.section{background:#ffffff;border:1px solid #cccccc;padding:25px;margin-bottom:25px}')
            html.append('.section h2{font-size:20px;color:#000000;font-weight:bold;margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid #2e5c8a}')

            # Combined Detection Statistics section (Chart & Mini Stats Cards)
            html.append('.detection-stats-combined{display:grid;grid-template-columns:1fr 1fr;gap:25px;align-items:stretch}')
            html.append('.detection-stats-combined .chart-side{display:flex;flex-direction:column;align-items:center;justify-content:center;padding-bottom:15px}')
            html.append('.detection-stats-combined .chart-side canvas{max-width:100%}')
            html.append('.detection-stats-combined .cards-side{display:grid;grid-template-columns:1fr 1fr;gap:15px;padding-right:10px}')
            html.append('.detection-stats-combined .stat-column{display:flex;flex-direction:column;gap:15px}')

            # Mini Stats Cards
            html.append('.mini-stat{background:#f8f8f8;border:1px solid #dddddd;padding:18px;text-align:center;transition:background 0.2s}')
            html.append('.mini-stat:hover{background:#f0f0f0}')
            html.append('.mini-stat h4{font-size:12px;color:#555555;font-weight:bold;text-transform:uppercase;margin-bottom:10px;letter-spacing:0.5px}')
            html.append('.mini-stat .count{font-size:32px;font-weight:bold;color:#2e5c8a;margin-bottom:5px}')
            html.append('.mini-stat .label{font-size:12px;color:#777777}')
            
            # Table styling 
            html.append('table{width:100%;border-collapse:collapse;margin-top:15px;font-size:14px}')
            html.append('th{background:#e8e8e8;border:1px solid #cccccc;padding:14px 15px;text-align:left;font-weight:bold;color:#000000;font-size:13px}')
            html.append('td{border:1px solid #dddddd;padding:12px 15px;background:#ffffff;font-size:14px}')
            html.append('tr:nth-child(even) td{background:#f9f9f9}')
            html.append('tr:hover td{background:#f0f0f0}')
            
            # Severity colors
            html.append('.severity-high{color:#cc0000;font-weight:bold}')
            html.append('.severity-medium{color:#cc6600;font-weight:bold}')
            html.append('.severity-low{color:#006600;font-weight:bold}')
            
            # Recommended Next Steps section
            html.append('.recommended-next-steps{background:#f8f8f8;border:1px solid #dddddd;padding:22px;margin-top:25px}')
            html.append('.recommended-next-steps h2{font-size:20px;color:#000000;font-weight:bold;margin-bottom:15px}')
            html.append('.recommended-next-steps ul{list-style:none;padding:0}')
            html.append('.recommended-next-steps li{padding:12px 15px;margin-bottom:8px;background:#ffffff;border:1px solid #dddddd;font-size:14px}')
            html.append('.recommended-next-steps li strong{color:#2e5c8a;font-weight:bold}')
            
            # Footer
            html.append('.footer{background:#f0f0f0;border:1px solid #cccccc;padding:25px;text-align:center;color:#555555;font-size:13px;margin-top:30px}')
            html.append('.footer p{margin:5px 0;font-size:14px}')
            html.append('.footer strong{color:#000000;font-size:15px}')
            
            # Indicator Types & Explanations
            html.append('.indicator-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid #cccccc;margin-bottom:25px}')
            html.append('.indicator-left{border-right:1px solid #cccccc;padding:20px;background:#fafafa}')
            html.append('.indicator-right{padding:20px;background:#fafafa}')
            html.append('.indicator-left h3,.indicator-right h3{font-size:15px;font-weight:bold;color:#000000;padding:14px 18px;background:#e8e8e8;border-bottom:1px solid #cccccc;margin-bottom:0}')
            html.append('.indicator-left table{width:100%;border-collapse:collapse;border:1px solid #e0e0e0;margin-top:14px}')
            html.append('.indicator-left th{background:#f0f0f0;border-bottom:1px solid #cccccc;border-right:1px solid #e0e0e0;padding:10px 14px;text-align:left;font-weight:bold;color:#000;font-size:13px}')
            html.append('.indicator-left td{border-bottom:1px solid #eeeeee;border-right:1px solid #eeeeee;padding:10px 14px;font-size:13px;background:#ffffff}')
            html.append('.indicator-left tr:nth-child(even) td{background:#f9f9f9}')
            html.append('.indicator-left tr:hover td{background:#f0f5ff}')
            html.append('.explanation-item{margin-bottom:14px;padding:12px 14px;background:#ffffff;border:1px solid #e0e0e0;border-left:4px solid #2e5c8a}')
            html.append('.explanation-title{font-size:12px;font-weight:bold;color:#2e5c8a;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:5px}')
            html.append('.indicator-disclaimer{font-size:12px;color:#BC0006;font-weight:normal;margin-top:20px;font-style:italic}')
            html.append('.explanation-text{font-size:13px;color:#333333;line-height:1.5}')

            # Paginated Table controls
            html.append('.pagination-info{font-size:13px;color:#555;margin-bottom:8px}')
            html.append('.pagination-controls{display:flex;align-items:center;gap:6px;margin-top:14px;flex-wrap:wrap}')
            html.append('.page-btn{background:#ffffff;border:1px solid #cccccc;color:#2e5c8a;padding:6px 12px;font-size:13px;cursor:pointer;transition:background 0.15s}')
            html.append('.page-btn:hover{background:#e8eef5}')
            html.append('.page-btn.active{background:#2e5c8a;color:#ffffff;border-color:#2e5c8a;font-weight:bold}')
            html.append('.page-btn:disabled{color:#aaaaaa;cursor:not-allowed;background:#f5f5f5}')
            html.append('.page-size-select{border:1px solid #cccccc;padding:6px 10px;font-size:13px;color:#333;background:#fff;cursor:pointer}')

            # Tooltip
            html.append('.tooltip{position:fixed;z-index:9999;background:#333333;color:#ffffff;padding:8px 12px;border-radius:3px;font-size:13px;pointer-events:none;opacity:0;transition:opacity 0.2s;box-shadow:0 2px 6px rgba(0,0,0,0.3)}')

            # Responsive design
            html.append('@media (max-width:900px){.detection-stats-combined{grid-template-columns:1fr}.detection-stats-combined .cards-side{grid-template-columns:repeat(2,1fr)}.indicator-grid{grid-template-columns:1fr}.indicator-left{border-right:none;border-bottom:1px solid #cccccc}}')
            html.append('@media (max-width:768px){body{padding:15px}.stats-grid{grid-template-columns:1fr}.header h1{font-size:28px}.stat-card .value{font-size:32px}.detection-stats-combined .cards-side{grid-template-columns:1fr}}')
            html.append('</style>')
            html.append('</head>')
            html.append('<body>')
            html.append('<div class="container">')
            
            # Header
            html.append('<div class="header">')
            html.append('<h1>NTFS Timestomping Detection Report</h1>')
            html.append('<p class="subtitle">Machine Learning Analysis of NTFS Timestamp Manipulation | An HTML Report visualization of the summary.txt | Generated: {0}</p>'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            html.append('<p class="disclaimer">DISCLAIMER:The NTFS Timestomping Detector Autopsy Ingest Module integrates a Machine Learning (ML) model to support analysis and decision-making. While these technologies enhance efficiency and provide valuable insights, they cannot fully replace human judgment. Final analysis, interpretations, and decisions should always be validated by qualified professionals.</p>')
            html.append('</div>')
            
            # Metadata section
            html.append('<div class="metadata">')
            html.append('<div class="metadata-grid">')
            html.append('<div class="metadata-item"><span class="metadata-label">Analysis Date</span><span class="metadata-value">{0}</span></div>'.format(esc(data['analysis_date'])))
            html.append('<div class="metadata-item"><span class="metadata-label">Model Used</span><span class="metadata-value">{0}</span></div>'.format(esc(data['model_used'])))
            html.append('<div class="metadata-item"><span class="metadata-label">Detection Threshold</span><span class="metadata-value">{0}</span></div>'.format(esc(data['threshold'])))
            html.append('<div class="metadata-item"><span class="metadata-label">Features Analyzed</span><span class="metadata-value">{0} behavioral features</span></div>'.format(data['methodology_features']))
            html.append('</div>')
            html.append('</div>')
            
            # Summary Stats
            html.append('<div class="stats-grid">')
            html.append('<div class="stat-card"><h3>Total Files Analyzed</h3><div class="value">{0:,}</div><div class="subtext">NTFS files processed</div></div>'.format(data['total_files']))
            html.append('<div class="stat-card"><h3>Files Flagged</h3><div class="value">{0:,}</div><div class="subtext">{1:.2f}% flagged as suspicious</div></div>'.format(data['files_flagged'], data['flag_percentage']))
            html.append('<div class="stat-card"><h3>Files Cleared</h3><div class="value">{0:,}</div><div class="subtext">{1:.2f}% determined benign</div></div>'.format(files_cleared, 100 - data['flag_percentage']))
            html.append('<div class="stat-card"><h3>Candidate Reduction</h3><div class="value">{0:.1f}%</div><div class="subtext">Analyst workload reduction</div></div>'.format(data['crr_percentage']))
            html.append('<div class="stat-card"><h3>Investigation Efficiency</h3><div class="value">{0:.1f}</div><div class="subtext">Files per true positive (NNI)</div></div>'.format(data['nni']))
            html.append('</div>')
            
            # Generated Output Files
            if data['output_files']:
                html.append('<div class="output-files">')
                html.append('<h2>Generated Output Files</h2>')
                html.append('<ul>')
                for output_file in data['output_files']:
                    file_path = output_file.get('path', output_file.get('full_path', ''))
                    file_name = output_file.get('name', '')
                    file_desc = output_file.get('description', '')
                    
                    if file_path and file_name:
                        html.append('<li><a href="file:///{0}"><strong>{1}</strong></a> - {2}</li>'.format(
                            esc(file_path.replace("\\", "/")),
                            esc(file_name),
                            esc(file_desc)
                        ))
                html.append('</ul>')
                html.append('</div>')
            
            
            # Indicator Types & Explanations
            indicator_definitions = [
                ('High Burstiness [>0.5]', 'Rapid successive timestamp modifications'),
                ('LogFile/USN Mismatch', 'Journal artifacts inconsistent with $MFT'),
                ('Backward Timestamp Jumps', 'Files where timestamps moved backwards'),
                ('Zero Nanosecond Precision', 'Files with suspiciously round timestamps'),
                ('SI-Only Modifications', '$STANDARD_INFORMATION modified, $FILE_NAME unchanged'),
            ]
            indicator_explanations = [
                ('HIGH_BURSTINESS (MEDIUM-HIGH Severity)',
                 'Multiple timestamp changes in rapid succession suggest automated or scripted manipulation. Normal file usage produces distributed timestamp changes.'),
                ('ARTIFACT_MISMATCH/LOGFILE/USN_MISMATCH (HIGH Severity)',
                 '$LogFile and $UsnJrnl independently record operations. Inconsistency with $MFT timestamps indicates post-operation modification. Journal entries are difficult to modify without detection.'),
                ('BACKWARD_TIMESTAMP (HIGH Severity when >1 day)',
                 'Timestamps that move backwards in time cannot occur through normal file operations. This definitively indicates deliberate manipulation to make files appear older than they actually are.'),
                ('ZERO_NANOSECONDS (MEDIUM Severity)',
                 'NTFS stores 100-nanosecond precision. Legitimate operations produce non-zero values. Zero nanoseconds indicates programmatic timestamp setting, typically via timestomping tools that don\'t populate sub-second precision.'),
                ('SI_ONLY_MODIFICATION (HIGH Severity)',
                 'Normal operations update both $STANDARD_INFORMATION and $FILE_NAME attributes. When only $SI is modified, it indicates tools that bypass the file system. The $FILE_NAME attribute retains the original timestamp.'),
            ]

            html.append('<div class="section">')
            html.append('<h2>Detection Indicators Distribution</h2>')

            # Bar Chart of Indicator Distribution
            html.append('<div class="chart-card full-width" style="border:none;padding:0;margin-bottom:20px">')
            html.append('<canvas id="indicatorsChart" width="1200" height="350"></canvas>')
            html.append('<div class="chart-note" style="text-align:center;margin-top:12px;font-size:14px;color:#000000">Number of files exhibiting each type of timestomping indicator</div>')
            html.append('</div>')

            # Indicator Types & Explanations
            html.append('<div class="indicator-grid">')

            # Indicator Types
            html.append('<div class="indicator-left">')
            html.append('<h3>Indicator Types and their Description</h3>')
            html.append('<table>')
            html.append('<thead><tr><th>Indicator Type</th><th>Description</th></tr></thead>')
            html.append('<tbody>')
            for ind_type, ind_desc in indicator_definitions:
                html.append('<tr><td><strong>{0}</strong></td><td>{1}</td></tr>'.format(esc(ind_type), esc(ind_desc)))
            html.append('</tbody></table>')
            html.append('<p class="indicator-disclaimer">DISCLAIMER: The indicator types, descriptions, and explanations provided are intended to support forensic analysis, not replace human judgment. Final interpretations of these specific timestamp anomalies must be validated by a qualified digital forensic professional.</p>')
            html.append('</div>')

            # Indicator Explanations
            html.append('<div class="indicator-right">')
            html.append('<h3>Indicator Explanations</h3>')
            html.append('<div style="padding:14px 0 0 0">')
            for exp_title, exp_text in indicator_explanations:
                html.append('<div class="explanation-item">')
                html.append('<div class="explanation-title">{0}</div>'.format(esc(exp_title)))
                html.append('<div class="explanation-text">{0}</div>'.format(esc(exp_text)))
                html.append('</div>')
            html.append('</div>')
            html.append('</div>')

            html.append('</div>') 
            html.append('</div>') 

            # Combined Detection Statistics Section (Severity Bar Chart & Mini Cards)
            html.append('<div class="section">')
            html.append('<h2>Detection Statistics</h2>')
            html.append('<div class="detection-stats-combined">')

            # Severity Distribution Bar Chart
            html.append('<div class="chart-side">')
            html.append('<canvas id="severityBarChart" width="480" height="360"></canvas>')
            html.append('<div class="chart-note" style="text-align:center;margin-top:12px;font-size:14px;color:#000000">Distribution of flagged files across severity levels <br><br>Severity reflects the strength and number of detection indicators, not a definitive determination of timestomping. All flagged files should be reviewed by a qualified digital forensic professional.</div>')
            html.append('</div>')

            # Mini Cards Statistics
            html.append('<div class="cards-side">')

            # Confidence Level
            html.append('<div class="stat-column">')
            html.append('<div class="mini-stat"><h4>High Confidence</h4><div class="count">{0}</div><div class="label">&gt; 0.5 score</div></div>'.format(data['high_confidence']))
            html.append('<div class="mini-stat"><h4>Medium Confidence</h4><div class="count">{0}</div><div class="label">0.1 - 0.5 score</div></div>'.format(data['medium_confidence']))
            html.append('<div class="mini-stat"><h4>Low Confidence</h4><div class="count">{0}</div><div class="label">{1} - 0.1 score</div></div>'.format(data['low_confidence'], esc(data['threshold'])))
            html.append('</div>')

            # Severity
            html.append('<div class="stat-column">')
            html.append('<div class="mini-stat"><h4>High Severity</h4><div class="count">{0}</div><div class="label">Immediate action</div></div>'.format(data['high_severity']))
            html.append('<div class="mini-stat"><h4>Medium Severity</h4><div class="count">{0}</div><div class="label">Manual review</div></div>'.format(data['medium_severity']))
            html.append('<div class="mini-stat"><h4>Low Severity</h4><div class="count">{0}</div><div class="label">Contextual review</div></div>'.format(data['low_severity']))
            html.append('</div>')

            html.append('</div>')
            html.append('</div>')  
            html.append('</div>')

            # List of All Suspicious Files Detected - Paginated Table
            html.append('<div class="section">')
            html.append('<h2>List of All Suspicious Files Detected</h2>')
            html.append('<div class="pagination-info" id="paginationInfo"></div>')
            html.append('<table id="filesTable">')
            html.append('<thead><tr><th>Rank</th><th>Confidence</th><th>Severity</th><th>Indicators</th><th>File Name</th></tr></thead>')
            html.append('<tbody id="filesTableBody">')
            if data['all_flagged_files']:
                for file_data in data['all_flagged_files']:
                    severity_class = 'severity-' + file_data['severity'].lower()
                    html.append('<tr class="file-row" data-rank="{0}"><td>{0}</td><td>{1:.6f}</td><td class="{2}">{3}</td><td>{4}</td><td>{5}</td></tr>'.format(
                        file_data['rank'],
                        file_data['confidence'],
                        severity_class,
                        esc(file_data['severity']),
                        file_data['indicators'],
                        esc(file_data['filename'])
                    ))
            else:
                html.append('<tr><td colspan="5" style="text-align:center;color:#999;font-style:italic;">No suspicious files data available</td></tr>')
            html.append('</tbody></table>')
            html.append('<div class="pagination-controls" id="paginationControls">')
            html.append('<label style="font-size:13px;color:#555">Rows per page:')
            html.append('<select class="page-size-select" id="pageSizeSelect" style="margin-left:6px">')
            html.append('<option value="10" selected>10</option><option value="25">25</option><option value="50">50</option><option value="100">100</option>')
            html.append('</select></label>')
            html.append('<span id="paginationButtons" style="display:flex;gap:6px;flex-wrap:wrap"></span>')
            html.append('</div>')
            html.append('</div>')
            
            # Recommended Next Steps
            if data['recommended_next_steps']:
                html.append('<div class="recommended-next-steps">')
                html.append('<h2>Recommended Next Steps</h2>')
                html.append('<ul>')
                for idx, next_step in enumerate(data['recommended_next_steps'], 1):
                    html.append('<li><strong>{0}.</strong> {1}</li>'.format(idx, esc(next_step)))
                html.append('</ul>')
                html.append('</div>')

            # Footer
            html.append('<div class="footer">')
            html.append('<p><strong>Digital Detectives Timestomping Detector v1.0</strong></p>')
            html.append('<p>Based on Oh et al. (2024) NTFS timestomping detection methodology</p>')
            html.append('<p>Model trained on 22 datasets with 52 known timestomped files from real APT campaigns</p>')
            html.append('</div>')
            
            html.append('</div>')
            
            # Inline JavaScript for charts
            html.append('<script>')
            html.append('(function(){')
            
            # Utility functions
            html.append('function $(id){return document.getElementById(id)}')
            html.append('function makeTip(){var t=document.createElement("div");t.className="tooltip";document.body.appendChild(t);return t}')
            html.append('var tip=makeTip();')
            html.append('function showTip(text,x,y){tip.textContent=text;tip.style.left=x+"px";tip.style.top=(y-30)+"px";tip.style.opacity=1}')
            html.append('function hideTip(){tip.style.opacity=0}')
            html.append('function within(mx,my,x,y,w,h){return mx>=x&&mx<=x+w&&my>=y&&my<=y+h}')
            
            # Bar Chart function for Detection Indicators Distribution
            html.append('function drawBar(id,labels,data){var c=$(id);if(!c)return;var ctx=c.getContext("2d");var w=c.width;var h=c.height;var pad=60;var max=Math.max(1,Math.max.apply(null,data));var barW=(w-pad*2)/Math.max(1,data.length);ctx.clearRect(0,0,w,h);ctx.font="12px Arial";ctx.textAlign="center";ctx.textBaseline="top";var meta=[];for(var i=0;i<data.length;i++){var val=data[i];var x=pad+i*barW+barW*0.12;var bw=barW*0.76;var bh=(h-pad*2)*(val/max);var y=h-pad-bh;ctx.fillStyle="#2e5c8a";ctx.fillRect(x,y,bw,bh);ctx.fillStyle="#000000";ctx.font="bold 12px Arial";ctx.textBaseline="bottom";ctx.fillText(String(val),x+bw/2,y-5);var lbl=labels[i]||"";ctx.fillStyle="#333333";ctx.font="11px Arial";ctx.textBaseline="top";ctx.save();ctx.translate(x+bw/2,h-pad+6);ctx.rotate(-0.5);ctx.fillText(lbl,0,0);ctx.restore();meta.push({x:x,y:y,w:bw,h:bh,label:lbl,val:val})}c.onmousemove=function(ev){var r=c.getBoundingClientRect();var mx=ev.clientX-r.left;var my=ev.clientY-r.top;var hit=null;for(var j=0;j<meta.length;j++){if(within(mx,my,meta[j].x,meta[j].y,meta[j].w,meta[j].h)){hit=meta[j];break}}if(hit){showTip(hit.label+": "+hit.val,ev.clientX,ev.clientY)}else{hideTip()}};c.onmouseleave=hideTip}')
            
            # Bar Chart function for Severity Statistics
            html.append('function drawBar(id,labels,data,colors){var c=$(id);if(!c)return;var ctx=c.getContext("2d");var w=c.width;var h=c.height;var pad=60;var max=Math.max(1,Math.max.apply(null,data));var barW=(w-pad*2)/Math.max(1,data.length);ctx.clearRect(0,0,w,h);ctx.font="12px Arial";ctx.textAlign="center";ctx.textBaseline="top";var meta=[];for(var i=0;i<data.length;i++){var val=data[i];var x=pad+i*barW+barW*0.12;var bw=barW*0.76;var bh=(h-pad*2)*(val/max);var y=h-pad-bh;ctx.fillStyle=(colors&&colors[i])?colors[i]:"#2e5c8a";ctx.fillRect(x,y,bw,bh);ctx.fillStyle="#000000";ctx.font="bold 12px Arial";ctx.textBaseline="bottom";ctx.fillText(String(val),x+bw/2,y-5);var lbl=labels[i]||"";ctx.fillStyle="#333333";ctx.font="11px Arial";ctx.textBaseline="top";ctx.save();ctx.translate(x+bw/2,h-pad+6);ctx.rotate(-0.5);ctx.fillText(lbl,0,0);ctx.restore();meta.push({x:x,y:y,w:bw,h:bh,label:lbl,val:val})}c.onmousemove=function(ev){var r=c.getBoundingClientRect();var mx=ev.clientX-r.left;var my=ev.clientY-r.top;var hit=null;for(var j=0;j<meta.length;j++){if(within(mx,my,meta[j].x,meta[j].y,meta[j].w,meta[j].h)){hit=meta[j];break}}if(hit){showTip(hit.label+": "+hit.val,ev.clientX,ev.clientY)}else{hideTip()}};c.onmouseleave=hideTip}')

            # Data for charts
            html.append('var indicatorLabels=' + js_array_str(['"' + esc(l).replace('"', '\\"') + '"' for l in indicator_labels]) + ';')
            html.append('var indicatorValues=' + js_array_str([str(v) for v in indicator_values]) + ';')
            html.append('var severityLabels=["Low Severity","Medium Severity","High Severity"];')
            html.append('var severityValues=[{0},{1},{2}];'.format(data['low_severity'], data['medium_severity'], data['high_severity']))
            html.append('var severityColors=["#006600","#cc6600","#cc0000"];')

            # Pagination JavaScript
            html.append('''
function initPagination(){
  var rows=Array.prototype.slice.call(document.querySelectorAll(".file-row"));
  if(!rows.length)return;
  var pageSize=10;var currentPage=1;
  var totalRows=rows.length;

  function getTotalPages(){return Math.max(1,Math.ceil(totalRows/pageSize));}

  function renderPage(page){
    currentPage=page;
    var start=(page-1)*pageSize;
    var end=start+pageSize;
    rows.forEach(function(r,i){r.style.display=(i>=start&&i<end)?"":"none";});
    var showing=Math.min(end,totalRows);
    var info=document.getElementById("paginationInfo");
    if(info)info.textContent="Showing "+(start+1)+" to "+showing+" of "+totalRows+" entries";
    renderButtons();
  }

  function renderButtons(){
    var container=document.getElementById("paginationButtons");
    if(!container)return;
    container.innerHTML="";
    var total=getTotalPages();

    function makeBtn(label,page,disabled,active){
      var b=document.createElement("button");
      b.textContent=label;
      b.className="page-btn"+(active?" active":"");
      b.disabled=disabled;
      if(!disabled)b.onclick=function(){renderPage(page);};
      container.appendChild(b);
    }

    makeBtn("Previous",currentPage-1,currentPage===1,false);

    var start=Math.max(1,currentPage-2);
    var end=Math.min(total,currentPage+2);
    if(start>1){makeBtn("1",1,false,false);if(start>2){var e=document.createElement("span");e.textContent="...";e.style.cssText="padding:6px 4px;font-size:13px;color:#555";container.appendChild(e);}}
    for(var p=start;p<=end;p++){makeBtn(String(p),p,false,p===currentPage);}
    if(end<total){if(end<total-1){var e2=document.createElement("span");e2.textContent="...";e2.style.cssText="padding:6px 4px;font-size:13px;color:#555";container.appendChild(e2);}makeBtn(String(total),total,false,false);}

    makeBtn("Next",currentPage+1,currentPage===total,false);
  }

  var sel=document.getElementById("pageSizeSelect");
  if(sel){sel.onchange=function(){pageSize=parseInt(this.value,10);renderPage(1);};}
  renderPage(1);
}
''')

            # Draw charts
            html.append('function ready(fn){if(document.readyState!=="loading"){fn()}else{document.addEventListener("DOMContentLoaded",fn)}}')
            html.append('ready(function(){')
            html.append('drawBar("indicatorsChart",indicatorLabels,indicatorValues);')
            html.append('drawBar("severityBarChart",severityLabels,severityValues,severityColors);')

            html.append('initPagination();')
            html.append('});')
            
            html.append('})();')
            html.append('</script>')
            html.append('</body>')
            html.append('</html>')
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(html))
            
            self.log(Level.INFO, "HTML report generated successfully")
            self.log(Level.INFO, "Report saved to: " + output_path)
            
            return output_path
            
        except Exception as e:
            self.log(Level.SEVERE, "Error generating HTML report: " + str(e))
            raise
    
    def generate_report_from_summary(self, summary_path, output_path):
        """
        Main method to generate HTML report from summary.txt file.
        """
        try:
            self.log(Level.INFO, "Starting HTML report generation")
            self.log(Level.INFO, "Input: " + summary_path)
            self.log(Level.INFO, "Output: " + output_path)
            
            data = self.parse_summary_file(summary_path)
            
            report_path = self.generate_html_report(data, output_path)
            
            if self.module and self.module_name:
                try:
                    self.module.currentCase.addReport(report_path, self.module_name, 'NTFS Timestomping Detection Report')
                    self.log(Level.INFO, 'Report registered in Autopsy Reports tree: ' + report_path)
                except Exception as e:
                    self.log(Level.INFO, 'Unable to register report: ' + str(e))
            else:
                self.log(Level.WARNING, "Module reference not available - report not registered in Autopsy Reports tree")
            
            self.log(Level.INFO, "HTML report generation completed successfully")
            
            return report_path
            
        except Exception as e:
            self.log(Level.SEVERE, "Failed to generate HTML report: " + str(e))
            raise

def generate_html_report(summary_path, output_path, logger_obj=None):
    generator = HTMLReportGenerator(logger_obj)
    return generator.generate_report_from_summary(summary_path, output_path)