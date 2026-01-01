# Digital Detectives - NTFS Timestomping Detection

**Machine Learning-Based Detection of Timestamp Manipulation in NTFS File Systems**

This repository contains a thesis project developing machine learning models to detect timestamp manipulation (timestomping) in NTFS filesystems using $LogFile and $UsnJrnl artifacts, based on Oh et al. (2024) methodology.

---

## Project Overview

### Research Objective

Develop an ML-based system to automatically detect timestamp manipulation in NTFS filesystems by analyzing cross-artifact patterns in $LogFile and $UsnJrnl transaction logs.

### Base Methodology

**Oh, Lee, and Hwang (2024)** - "Forensic Detection of Timestamp Manipulation for Digital Forensic Investigation"  
Published in IEEE Access ([DOI: 10.1109/ACCESS.2024.10517044](https://ieeexplore.ieee.org/document/10517044))

### Detection Approach

1. **Event-level data merge**: Preserve LSN/USN identifiers for exact ground truth matching
2. **Forensic pattern extraction**: Zero nanoseconds, time reversal events, cross-artifact validation
3. **ML classification**: Train models on forensic features to detect timestomped files
4. **Production deployment**: Autopsy integration for operational forensic investigations

---

## Dataset Structure

### Training Datasets (19 total)

**PE Cases (12)**: 01-PE through 12-PE  
**APT Cases (7)**: 01-APT17, 02-APT19, 03-APT21, 04-APT28, 05-APT29, 06-APT30, 07-APT37

Each dataset contains:
- **LogFile CSV**: NTFS transaction log with timestamp change events
- **UsnJrnl CSV**: NTFS change journal with file modification events  
- **Suspicious CSV**: Ground truth labels from Oh et al.'s NTFS Artifact Analysis Tool

### Validation Datasets (5 total)

**Lone Wolf**: 12 timestomped files (Autopsy integration validation)  
**09-APT40**: 1 file (zero nanoseconds + file move)  
**12-Kimusky**: 3 files (zero nanoseconds)  
**13-Winnti731**: 1 file (zero nanoseconds + file move)  
**02-APT19**: 1 file (zero nanoseconds + file move)

**Total**: 18 validation files, all with "Zero in 100-nanoseconds" pattern

**Held-out datasets**: 08-APT38, 10-DarkHotel663, 11-DarkHotelbbd, 14-Winnti43b (moved to training for increased dataset size)