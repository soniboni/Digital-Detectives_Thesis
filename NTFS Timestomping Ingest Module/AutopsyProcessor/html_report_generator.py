# -*- coding: utf-8 -*-
"""
HTML Report Generator for NTFS Timestomping Detection (Jython 2.7 Compatible)

Creates comprehensive HTML summary reports with statistics and visualizations.
No external dependencies - all charting is done with inline JavaScript.
Follows Autopsy report styling conventions.
"""

import os
import re
from datetime import datetime
from java.util.logging import Level


class HTMLReportGenerator:
    """Generates comprehensive HTML reports for NTFS timestomping detection analysis"""
    
    def __init__(self, logger_obj=None, module=None, module_name=None):
        """
        Initialize HTML Report Generator.
        
        Args:
            logger_obj: Autopsy Logger object for logging
            module: Autopsy ingest module instance (for report registration)
            module_name: Name of the module for report registration
        """
        self.logger = logger_obj
        self.module = module
        self.module_name = module_name
    
    def log(self, level, msg):
        """Log message using Autopsy logger or print."""
        if self.logger:
            self.logger.logp(level, self.__class__.__name__, "log", msg)
        else:
            print("[{0}] {1}".format(level, msg))
    
    def parse_summary_file(self, summary_path):
        """
        Parse summary.txt file to extract report data.
        
        Args:
            summary_path: Path to summary.txt file
            
        Returns:
            dict: Parsed report data
        """
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
                'top_10_files': [],
                'indicators': {},
                'methodology_features': 0,
                'output_files': []
            }
            
            # Extract basic information
            match = re.search(r'Analysis Date:\s+(.+)', content)
            if match:
                data['analysis_date'] = match.group(1).strip()
            
            match = re.search(r'Model Used:\s+(.+)', content)
            if match:
                data['model_used'] = match.group(1).strip()
            
            match = re.search(r'Threshold:\s+(.+)', content)
            if match:
                data['threshold'] = match.group(1).strip()
            
            # Extract executive summary numbers
            match = re.search(r'Total Files Analyzed:\s+([\d,]+)', content)
            if match:
                data['total_files'] = int(match.group(1).replace(',', ''))
            
            match = re.search(r'Files Flagged:\s+([\d,]+)\s+\(([\d.]+)%\)', content)
            if match:
                data['files_flagged'] = int(match.group(1).replace(',', ''))
                data['flag_percentage'] = float(match.group(2))
            
            # Extract confidence breakdown
            match = re.search(r'High Confidence \(>0\.5\):\s+(\d+)', content)
            if match:
                data['high_confidence'] = int(match.group(1))
            
            match = re.search(r'Medium Confidence \(0\.1-0\.5\):\s+(\d+)', content)
            if match:
                data['medium_confidence'] = int(match.group(1))
            
            match = re.search(r'Low Confidence \([^)]+\):\s+(\d+)', content)
            if match:
                data['low_confidence'] = int(match.group(1))
            
            # Extract severity breakdown
            match = re.search(r'HIGH Severity:\s+(\d+)', content)
            if match:
                data['high_severity'] = int(match.group(1))
            
            match = re.search(r'MEDIUM Severity:\s+(\d+)', content)
            if match:
                data['medium_severity'] = int(match.group(1))
            
            match = re.search(r'LOW Severity:\s+(\d+)', content)
            if match:
                data['low_severity'] = int(match.group(1))
            
            # Extract metrics
            match = re.search(r'Candidate Reduction Rate:\s+([\d.]+)%', content)
            if match:
                data['crr_percentage'] = float(match.group(1))
            
            match = re.search(r'Number Needed to Investigate:\s+~([\d.]+)', content)
            if match:
                data['nni'] = float(match.group(1))
            
            # Extract top 10 files - FIXED: Use section markers to avoid crossing boundaries
            top_10_start = content.find('TOP 10 MOST SUSPICIOUS FILES')
            detection_summary_start = content.find('DETECTION INDICATOR SUMMARY')
            
            if top_10_start != -1 and detection_summary_start != -1:
                top_10_section = content[top_10_start:detection_summary_start]
                # Find the header line and data lines
                lines = top_10_section.split('\n')
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
                    
                    # Parse: Rank Confidence Severity Indicators FileName
                    parts = line.split(None, 4)
                    if len(parts) >= 5:
                        try:
                            data['top_10_files'].append({
                                'rank': int(parts[0]),
                                'confidence': float(parts[1]),
                                'severity': parts[2],
                                'indicators': int(parts[3]),
                                'filename': parts[4]
                            })
                        except (ValueError, IndexError) as e:
                            self.log(Level.WARNING, "Failed to parse top 10 line: " + line)
                            continue
           
            # Extract indicator counts - FIXED: Use section markers to avoid crossing boundaries
            indicators_start = content.find('DETECTION INDICATOR SUMMARY')
            explanations_start = content.find('INDICATOR EXPLANATIONS')
            
            if indicators_start != -1 and explanations_start != -1:
                indicators_section = content[indicators_start:explanations_start]
                # Find the header line
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
                    
                    # Parse: IndicatorName (multiple spaces) Count (multiple spaces) Description
                    match = re.match(r'(.+?)\s{2,}(\d+)\s{2,}(.+)', line)
                    if match:
                        indicator_name = match.group(1).strip()
                        indicator_count = int(match.group(2))
                        data['indicators'][indicator_name] = indicator_count
                        
                        self.log(Level.INFO, "Found indicator: {0} = {1}".format(indicator_name, indicator_count))

            # Extract methodology features count
            match = re.search(r'Features analyzed:\s+(\d+)', content)
            if match:
                data['methodology_features'] = int(match.group(1))
            
            # Extract output files
            output_section = re.search(r'OUTPUT FILES GENERATED.*?\n-+\n(.*?)\n\n', content, re.DOTALL)
            if output_section:
                lines = output_section.group(1).strip().split('\n')
                for line in lines:
                    match = re.match(r'\d+\.\s+(.+?)\s+-\s+(.+)', line)
                    if match:
                        data['output_files'].append({
                            'name': match.group(1).strip(),
                            'description': match.group(2).strip()
                        })
            
            self.log(Level.INFO, "Successfully parsed summary file")
            self.log(Level.INFO, "Total files: {0}, Flagged: {1}".format(data['total_files'], data['files_flagged']))
            self.log(Level.INFO, "Indicators found: {0}".format(len(data['indicators'])))
            self.log(Level.INFO, "Top 10 files found: {0}".format(len(data['top_10_files'])))

            return data
            
        except Exception as e:
            self.log(Level.SEVERE, "Error parsing summary file: " + str(e))
            raise
    
    def generate_html_report(self, data, output_path):
        """
        Generate HTML report with inline JavaScript visualizations.
        
        Args:
            data: Parsed summary data dictionary
            output_path: Path to save HTML report
            
        Returns:
            str: Path to generated HTML file
        """
        try:
            self.log(Level.INFO, "Generating HTML report at: " + output_path)
            
            # Helper function for JavaScript array formatting
            def js_array_str(values):
                return '[' + ','.join(values) + ']'
            
            # Helper function for HTML escaping
            def esc(x):
                try:
                    return str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
                except Exception:
                    return ''
            
            # Prepare indicator data for charts
            indicator_labels = []
            indicator_values = []
            for indicator_name, count in sorted(data['indicators'].items(), key=lambda x: x[1], reverse=True):
                short_name = indicator_name
                if len(short_name) > 35:
                    short_name = short_name[:32] + "..."

                indicator_labels.append(short_name)
                indicator_values.append(count)
            
            # Calculate files cleared
            files_cleared = data['total_files'] - data['files_flagged']
            
            # Build HTML content
            html = []
            html.append('<!DOCTYPE html>')
            html.append('<html lang="en">')
            html.append('<head>')
            html.append('<meta charset="utf-8"/>')
            html.append('<meta name="viewport" content="width=device-width, initial-scale=1"/>')
            html.append('<title>NTFS Timestomping Detection Report</title>')
            
            # CSS Styles - Autopsy-inspired design
            html.append('<style>')
            html.append('*{margin:0;padding:0;box-sizing:border-box}')
            html.append('body{font-family:Arial,Helvetica,sans-serif;background:#ffffff;color:#000000;line-height:1.7;padding:25px}')
            html.append('.container{max-width:1400px;margin:0 auto;background:#ffffff}')
            
            # Header - Autopsy style with blue accent
            html.append('.header{background:#2e5c8a;color:#ffffff;padding:30px 35px;margin-bottom:30px;border-bottom:3px solid #1a3a5a}')
            html.append('.header h1{font-size:32px;font-weight:bold;margin-bottom:10px;letter-spacing:0.3px}')
            html.append('.header .subtitle{font-size:16px;opacity:0.95;font-weight:normal}')

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

            # Charts
            html.append('.charts-grid{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:25px}')
            html.append('.chart-card{background:#ffffff;border:1px solid #cccccc;padding:20px;transition:box-shadow 0.2s}')
            html.append('.chart-card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.15)}')
            html.append('.chart-card h2{font-size:16px;color:#000000;font-weight:bold;margin-bottom:15px;padding-bottom:8px;border-bottom:2px solid #2e5c8a}')
            html.append('.chart-card canvas{max-width:100%;height:auto}')
            html.append('.chart-card .chart-note{font-size:11px;color:#666666;margin-top:10px;font-style:italic}')
            html.append('.full-width{grid-column:1/-1}')
            
            # Section styling
            html.append('.section{background:#ffffff;border:1px solid #cccccc;padding:25px;margin-bottom:25px}')
            html.append('.section h2{font-size:20px;color:#000000;font-weight:bold;margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid #2e5c8a}')

            # Combined Detection Statistics section (35% chart, 65% mini cards)
            html.append('.detection-stats-combined{display:grid;grid-template-columns:35% 65%;gap:25px;align-items:start;padding:0 20px}')
            html.append('.detection-stats-combined .chart-side{display:flex;flex-direction:column;align-items:center;padding-bottom:15px}')
            html.append('.detection-stats-combined .chart-side canvas{max-width:100%}')
            html.append('.detection-stats-combined .cards-side{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;padding-right:10px}')

            # Mini stats cards
            html.append('.mini-stat{background:#f8f8f8;border:1px solid #dddddd;padding:18px;text-align:center;transition:background 0.2s}')
            html.append('.mini-stat:hover{background:#f0f0f0}')
            html.append('.mini-stat h4{font-size:12px;color:#555555;font-weight:bold;text-transform:uppercase;margin-bottom:10px;letter-spacing:0.5px}')
            html.append('.mini-stat .count{font-size:32px;font-weight:bold;color:#2e5c8a;margin-bottom:5px}')
            html.append('.mini-stat .label{font-size:12px;color:#777777}')
            
            # Table styling - Autopsy standard with improved readability
            html.append('table{width:100%;border-collapse:collapse;margin-top:15px;font-size:14px}')
            html.append('th{background:#e8e8e8;border:1px solid #cccccc;padding:14px 15px;text-align:left;font-weight:bold;color:#000000;font-size:13px}')
            html.append('td{border:1px solid #dddddd;padding:12px 15px;background:#ffffff;font-size:14px}')
            html.append('tr:nth-child(even) td{background:#f9f9f9}')
            html.append('tr:hover td{background:#f0f0f0}')
            
            # Severity colors
            html.append('.severity-high{color:#cc0000;font-weight:bold}')
            html.append('.severity-medium{color:#cc6600;font-weight:bold}')
            html.append('.severity-low{color:#006600;font-weight:bold}')
            
            # Output files section
            html.append('.output-files{background:#f8f8f8;border:1px solid #dddddd;padding:22px;margin-top:25px}')
            html.append('.output-files h3{font-size:16px;color:#000000;font-weight:bold;margin-bottom:15px}')
            html.append('.output-files ul{list-style:none;padding:0}')
            html.append('.output-files li{padding:12px 15px;margin-bottom:8px;background:#ffffff;border:1px solid #dddddd;font-size:14px}')
            html.append('.output-files li strong{color:#2e5c8a;font-weight:bold}')
            
            # Footer
            html.append('.footer{background:#f0f0f0;border:1px solid #cccccc;padding:25px;text-align:center;color:#555555;font-size:13px;margin-top:30px}')
            html.append('.footer p{margin:5px 0;font-size:14px}')
            html.append('.footer strong{color:#000000;font-size:15px}')
            
            # Tooltip
            html.append('.tooltip{position:fixed;z-index:9999;background:#333333;color:#ffffff;padding:8px 12px;border-radius:3px;font-size:13px;pointer-events:none;opacity:0;transition:opacity 0.2s;box-shadow:0 2px 6px rgba(0,0,0,0.3)}')

            # Responsive design
            html.append('@media (max-width:900px){.detection-stats-combined{grid-template-columns:1fr}.detection-stats-combined .cards-side{grid-template-columns:repeat(2,1fr)}}')
            html.append('@media (max-width:768px){body{padding:15px}.stats-grid{grid-template-columns:1fr}.header h1{font-size:28px}.stat-card .value{font-size:32px}.detection-stats-combined .cards-side{grid-template-columns:1fr}}')
            html.append('</style>')
            html.append('</head>')
            html.append('<body>')
            html.append('<div class="container">')
            
            # Header
            html.append('<div class="header">')
            html.append('<h1>NTFS Timestomping Detection Report</h1>')
            html.append('<p class="subtitle">Machine Learning Analysis of NTFS Timestamp Manipulation | An HTML Report visualization of the summary.txt | Generated: {0}</p>'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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
            
            # Summary stats grid
            html.append('<div class="stats-grid">')
            html.append('<div class="stat-card"><h3>Total Files Analyzed</h3><div class="value">{0:,}</div><div class="subtext">NTFS files processed</div></div>'.format(data['total_files']))
            html.append('<div class="stat-card"><h3>Files Flagged</h3><div class="value">{0:,}</div><div class="subtext">{1:.2f}% flagged as suspicious</div></div>'.format(data['files_flagged'], data['flag_percentage']))
            html.append('<div class="stat-card"><h3>Files Cleared</h3><div class="value">{0:,}</div><div class="subtext">{1:.2f}% determined benign</div></div>'.format(files_cleared, 100 - data['flag_percentage']))
            html.append('<div class="stat-card"><h3>Candidate Reduction</h3><div class="value">{0:.1f}%</div><div class="subtext">Analyst workload reduction</div></div>'.format(data['crr_percentage']))
            html.append('<div class="stat-card"><h3>Investigation Efficiency</h3><div class="value">{0:.1f}</div><div class="subtext">Files per true positive (NNI)</div></div>'.format(data['nni']))
            html.append('</div>')
            
            # Charts section - Detection Indicators Bar Chart (full width)
            html.append('<div class="charts-grid">')
            html.append('<div class="chart-card full-width">')
            html.append('<h2>Detection Indicators Distribution</h2>')
            html.append('<canvas id="indicatorsChart" width="1200" height="350"></canvas>')
            html.append('<div class="chart-note">Number of files exhibiting each type of timestomping indicator</div>')
            html.append('</div>')
            html.append('</div>')
            
            # Combined Detection Statistics Section (Severity Doughnut + 6 Mini Cards)
            html.append('<div class="section">')
            html.append('<h2>Detection Statistics</h2>')
            html.append('<div class="detection-stats-combined">')
            
            # Left side - Severity Distribution Doughnut Chart (35%)
            html.append('<div class="chart-side">')
            html.append('<canvas id="severityChart" width="350" height="420"></canvas>')
            html.append('<div class="chart-note" style="margin-top:10px;text-align:center;">Severity Distribution</div>')
            html.append('</div>')
            
            # Right side - 6 Mini Cards (65%)
            html.append('<div class="cards-side">')
            html.append('<div class="mini-stat"><h4>High Confidence</h4><div class="count">{0}</div><div class="label">&gt; 0.5 score</div></div>'.format(data['high_confidence']))
            html.append('<div class="mini-stat"><h4>Medium Confidence</h4><div class="count">{0}</div><div class="label">0.1 - 0.5 score</div></div>'.format(data['medium_confidence']))
            html.append('<div class="mini-stat"><h4>Low Confidence</h4><div class="count">{0}</div><div class="label">{1} - 0.1 score</div></div>'.format(data['low_confidence'], data['threshold']))
            html.append('<div class="mini-stat"><h4>High Severity</h4><div class="count">{0}</div><div class="label">Immediate action</div></div>'.format(data['high_severity']))
            html.append('<div class="mini-stat"><h4>Medium Severity</h4><div class="count">{0}</div><div class="label">Manual review</div></div>'.format(data['medium_severity']))
            html.append('<div class="mini-stat"><h4>Low Severity</h4><div class="count">{0}</div><div class="label">Contextual review</div></div>'.format(data['low_severity']))
            html.append('</div>')
            
            html.append('</div>')  # Close detection-stats-combined
            html.append('</div>')  # Close section
            
            # Top 10 Files Table
            html.append('<div class="section">')
            html.append('<h2>Top 10 Most Suspicious Files</h2>')
            html.append('<table>')
            html.append('<thead><tr><th>Rank</th><th>Confidence</th><th>Severity</th><th>Indicators</th><th>File Name</th></tr></thead>')
            html.append('<tbody>')
            if data['top_10_files']:
                for file_data in data['top_10_files']:
                    severity_class = 'severity-' + file_data['severity'].lower()
                    html.append('<tr><td>{0}</td><td>{1:.6f}</td><td class="{2}">{3}</td><td>{4}</td><td>{5}</td></tr>'.format(
                        file_data['rank'],
                        file_data['confidence'],
                        severity_class,
                        esc(file_data['severity']),
                        file_data['indicators'],
                        esc(file_data['filename'])
                    ))
            else:
                html.append('<tr><td colspan="5" style="text-align:center;color:#999;font-style:italic;">No top 10 files data available</td></tr>')
            html.append('</tbody></table>')
            html.append('</div>')
            
            # Output Files
            if data['output_files']:
                html.append('<div class="output-files">')
                html.append('<h3>Generated Output Files</h3>')
                html.append('<ul>')
                for output_file in data['output_files']:
                    html.append('<li><strong>{0}</strong> - {1}</li>'.format(
                        esc(output_file['name']),
                        esc(output_file['description'])
                    ))
                html.append('</ul>')
                html.append('</div>')
            
            # Footer
            html.append('<div class="footer">')
            html.append('<p><strong>Digital Detectives Timestomping Detector v1.0</strong></p>')
            html.append('<p>Based on Oh et al. (2024) NTFS timestomping detection methodology</p>')
            html.append('<p>Model trained on 22 datasets with 52 known timestomped files from real APT campaigns</p>')
            html.append('</div>')
            
            html.append('</div>')  # Close container
            
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
            
            # Bar chart function
            html.append('function drawBar(id,labels,data){var c=$(id);if(!c)return;var ctx=c.getContext("2d");var w=c.width;var h=c.height;var pad=60;var max=Math.max(1,Math.max.apply(null,data));var barW=(w-pad*2)/Math.max(1,data.length);ctx.clearRect(0,0,w,h);ctx.font="12px Arial";ctx.textAlign="center";ctx.textBaseline="top";var meta=[];for(var i=0;i<data.length;i++){var val=data[i];var x=pad+i*barW+barW*0.12;var bw=barW*0.76;var bh=(h-pad*2)*(val/max);var y=h-pad-bh;ctx.fillStyle="#2e5c8a";ctx.fillRect(x,y,bw,bh);ctx.fillStyle="#000000";ctx.font="bold 12px Arial";ctx.textBaseline="bottom";ctx.fillText(String(val),x+bw/2,y-5);var lbl=labels[i]||"";ctx.fillStyle="#333333";ctx.font="11px Arial";ctx.textBaseline="top";ctx.save();ctx.translate(x+bw/2,h-pad+6);ctx.rotate(-0.5);ctx.fillText(lbl,0,0);ctx.restore();meta.push({x:x,y:y,w:bw,h:bh,label:lbl,val:val})}c.onmousemove=function(ev){var r=c.getBoundingClientRect();var mx=ev.clientX-r.left;var my=ev.clientY-r.top;var hit=null;for(var j=0;j<meta.length;j++){if(within(mx,my,meta[j].x,meta[j].y,meta[j].w,meta[j].h)){hit=meta[j];break}}if(hit){showTip(hit.label+": "+hit.val,ev.clientX,ev.clientY)}else{hideTip()}};c.onmouseleave=hideTip}')
            
            # Doughnut chart function
            html.append('function drawDoughnut(id,labels,data,colors){var c=$(id);if(!c)return;var ctx=c.getContext("2d");var w=c.width;var h=c.height;var cx=w/2;var cy=h/2-20;var r=Math.min(w,h)/2-50;var total=data.reduce(function(a,b){return a+b},0)||1;var start=-Math.PI/2;ctx.clearRect(0,0,w,h);var arcs=[];for(var i=0;i<data.length;i++){var val=data[i];var angle=2*Math.PI*(val/total);ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,start,start+angle);ctx.closePath();ctx.fillStyle=colors[i%colors.length];ctx.fill();ctx.strokeStyle="#ffffff";ctx.lineWidth=2;ctx.stroke();arcs.push({start:start,end:start+angle,val:val,label:labels[i],color:colors[i%colors.length]});start+=angle}var ir=r*0.58;ctx.globalCompositeOperation="destination-out";ctx.beginPath();ctx.arc(cx,cy,ir,0,2*Math.PI);ctx.fill();ctx.globalCompositeOperation="source-over";ctx.fillStyle="#000000";ctx.font="bold 14px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("Total",cx,cy-8);ctx.font="bold 20px Arial";ctx.fillText(total.toLocaleString(),cx,cy+12);var ly=cy+r+50;ctx.font="11px Arial";ctx.textAlign="left";for(var k=0;k<labels.length;k++){ctx.fillStyle=colors[k%colors.length];ctx.fillRect(cx-r,ly,10,10);ctx.fillStyle="#000000";ctx.fillText(labels[k]+" ("+data[k]+")",cx-r+15,ly+8);ly+=16}c.onmousemove=function(ev){var rct=c.getBoundingClientRect();var mx=ev.clientX-rct.left;var my=ev.clientY-rct.top;var dx=mx-cx;var dy=my-cy;var dist=Math.hypot(dx,dy);if(dist<ir||dist>r){hideTip();return}var ang=Math.atan2(dy,dx);while(ang<-Math.PI/2){ang+=Math.PI*2}var hit=null;for(var j=0;j<arcs.length;j++){if(ang>=arcs[j].start&&ang<=arcs[j].end){hit=arcs[j];break}}if(hit){var pct=((hit.val/total)*100).toFixed(1);showTip(hit.label+": "+hit.val+" ("+pct+"%)",ev.clientX,ev.clientY)}else{hideTip()}};c.onmouseleave=hideTip}')
            
            # Data for charts
            html.append('var indicatorLabels=' + js_array_str(['"' + esc(l).replace('"', '\\"') + '"' for l in indicator_labels]) + ';')
            html.append('var indicatorValues=' + js_array_str([str(v) for v in indicator_values]) + ';')
            html.append('var severityLabels=["HIGH Severity","MEDIUM Severity","LOW Severity"];')
            html.append('var severityValues=[{0},{1},{2}];'.format(data['high_severity'], data['medium_severity'], data['low_severity']))
            html.append('var severityColors=["#cc0000","#cc6600","#006600"];')
            
            # Ready function to draw charts
            html.append('function ready(fn){if(document.readyState!=="loading"){fn()}else{document.addEventListener("DOMContentLoaded",fn)}}')
            html.append('ready(function(){')
            html.append('drawBar("indicatorsChart",indicatorLabels,indicatorValues);')
            html.append('drawDoughnut("severityChart",severityLabels,severityValues,severityColors);')
            html.append('});')
            
            html.append('})();')
            html.append('</script>')
            html.append('</body>')
            html.append('</html>')
            
            # Write file
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
        
        Args:
            summary_path: Path to summary.txt file
            output_path: Path to save HTML report
            
        Returns:
            str: Path to generated HTML file
        """
        try:
            self.log(Level.INFO, "Starting HTML report generation")
            self.log(Level.INFO, "Input: " + summary_path)
            self.log(Level.INFO, "Output: " + output_path)
            
            # Parse summary file
            data = self.parse_summary_file(summary_path)
            
            # Generate HTML report
            report_path = self.generate_html_report(data, output_path)
            
            # Register report so it shows in Autopsy Reports tree
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
    """
    Convenience function to generate HTML report from summary file.
    
    Args:
        summary_path: Path to summary.txt file
        output_path: Path to save HTML report
        logger_obj: Optional Autopsy Logger object
        
    Returns:
        str: Path to generated HTML file
    """
    generator = HTMLReportGenerator(logger_obj)
    return generator.generate_report_from_summary(summary_path, output_path)