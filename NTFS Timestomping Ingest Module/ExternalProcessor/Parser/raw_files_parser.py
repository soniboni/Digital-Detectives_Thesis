# -*- coding: utf-8 -*-

"""
Raw Files Parser Module for NTFS Timestomping Detection

This module provides parsing functionality for NTFS system files ($MFT, $LogFile, $UsnJrnl:$J).
It converts binary system files into human-readable CSV structures for further analysis.

Environment: Python 3.x
Dependencies: dfir_ntfs, pandas, datetime, struct

Parser Classes:
    - MFTParser: Parses Master File Table ($MFT)
    - USNJournalParser: Parses Update Sequence Number Journal ($UsnJrnl:$J)
    - LogFileParser: Parses transaction log file ($LogFile)
    - RawFilesParser: Orchestrator class that coordinates all parsers
"""

import sys
import struct
import logging
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Third-party imports
from Parser.ThirdParty.dfir_ntfs import MFT
from Parser.ThirdParty.dfir_ntfs.USN import ChangeJournalParser, ResolveReasonCodes
from Parser.ThirdParty.dfir_ntfs.LogFile import LogFileParser


# Configure logging
logger = logging.getLogger(__name__)


class MFTParser:
    """
    Parser for Master File Table ($MFT).
    
    Extracts:
    - Standard Information ($SI) attributes: Creation, Modification, MFT Change, Access timestamps
    - File Name ($FN) attributes: Creation, Modification, MFT Change, Access timestamps
    - File metadata: Entry number, active status, parent FRN, file path
    """
    
    # NTFS attribute type codes
    STANDARD_INFORMATION = 0x10
    FILE_NAME = 0x30
    
    @staticmethod
    def mft_format_timestamp(dt_obj: Optional[datetime]) -> Optional[str]:
        """Format datetime object to string with microsecond precision."""
        if dt_obj is None:
            return None
        try:
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def mft_extract_si_timestamps(file_record) -> Dict[str, Optional[str]]:
        """
        Extract $STANDARD_INFORMATION timestamps from a file record.
        
        Returns dict with keys: SI_C (Creation), SI_M (Modification), 
                                SI_E (MFT Change), SI_A (Access)
        """
        si_timestamps = {
            "SI_C": None,
            "SI_M": None,
            "SI_E": None,
            "SI_A": None
        }
        
        try:
            for attribute in file_record.attributes():
                if attribute.type_code == MFTParser.STANDARD_INFORMATION:
                    try:
                        si_timestamps["SI_C"] = MFTParser.mft_format_timestamp(attribute.standard_information.get_creation_time())
                        si_timestamps["SI_M"] = MFTParser.mft_format_timestamp(attribute.standard_information.get_modification_time())
                        si_timestamps["SI_E"] = MFTParser.mft_format_timestamp(attribute.standard_information.get_mft_change_time())
                        si_timestamps["SI_A"] = MFTParser.mft_format_timestamp(attribute.standard_information.get_access_time())
                        break
                    except Exception as e:
                        logger.debug(f"Error extracting SI timestamps: {e}")
        except Exception as e:
            logger.debug(f"Error iterating SI attributes: {e}")
        
        return si_timestamps
    
    @staticmethod
    def mft_extract_fn_info(file_record, parser) -> Dict[str, Optional[str]]:
        """
        Extract $FILE_NAME attribute information including timestamps.
        
        Returns dict with keys: FileName, ParentFRN, FN_C, FN_M, FN_E, FN_A
        """
        fn_info = {
            "FileName": None,
            "ParentFRN": None,
            "FN_C": None,
            "FN_M": None,
            "FN_E": None,
            "FN_A": None
        }
        
        try:
            for attribute in file_record.attributes():
                if attribute.type_code == MFTParser.FILE_NAME:
                    try:
                        fn = attribute.value_decoded()
                        file_name = fn.get_file_name()
                        namespace = fn.get_flags()
                        
                        if fn_info["FileName"] is not None or namespace in (1,3):
                            fn_info["FileName"] = attribute.file_name.get_name()
                            fn_info["ParentFRN"] = attribute.file_name.get_parent_directory_reference() & 0xFFFFFFFFFFFF
                            fn_info["FN_C"] = MFTParser.mft_format_timestamp(attribute.file_name.get_creation_time())
                            fn_info["FN_M"] = MFTParser.mft_format_timestamp(attribute.file_name.get_modification_time())
                            fn_info["FN_E"] = MFTParser.mft_format_timestamp(attribute.file_name.get_mft_change_time())
                            fn_info["FN_A"] = MFTParser.mft_format_timestamp(attribute.file_name.get_access_time())

                            if namespace in (1,3):
                                break
                    except Exception as e:
                        logger.debug(f"Error extracting FN info: {e}")
        except Exception as e:
            logger.debug(f"Error iterating FN attributes: {e}")
        
        return fn_info
    
    @staticmethod
    def mft_build_file_path(file_record, parser, path_cache: Dict) -> Optional[str]:
        """Build the full file path for a given file record."""
        try:
            paths = parser.build_full_paths(file_record)
            if paths:
                return paths[0][0] if isinstance(paths[0], tuple) else paths[0]
        except Exception as e:
            logger.debug(f"Error building file path: {e}")
        return None
    
    @classmethod
    def parse_mft(cls, mft_path: Path) -> pd.DataFrame:
        """
        Parse the MFT file and extract all relevant metadata.
        
        Args:
            mft_path: Path to the $MFT file
            
        Returns:
            pandas.DataFrame: Parsed MFT data with columns for all timestamps and metadata
        """
        records = []
        path_cache = {}
        
        logger.info(f"Parsing MFT file: {mft_path.name}")
        
        with open(mft_path, "rb") as mft_file:
            parser = MFT.MasterFileTableParser(mft_file)
            
            error_count = 0
            
            for file_record in parser.file_records():
                try:
                    
                    entry_number = file_record.get_master_file_table_number()
                    is_active = file_record.is_in_use()
                    lsn = file_record.get_logfile_sequence_number()
                    
                    si_ts = cls.mft_extract_si_timestamps(file_record)
                    fn_info = cls.mft_extract_fn_info(file_record, parser)
                    file_path = cls.mft_build_file_path(file_record, parser, path_cache)
                    
                    record = {
                        "EntryNumber": entry_number,
                        "FileName": fn_info["FileName"],
                        "FilePath": file_path,
                        "IsActive": is_active,
                        "LSN": lsn,
                        "ParentFRN": fn_info["ParentFRN"],
                        "$SI-C": si_ts["SI_C"],
                        "$SI-M": si_ts["SI_M"],
                        "$SI-E": si_ts["SI_E"],
                        "$SI-A": si_ts["SI_A"],
                        "$FN-C": fn_info["FN_C"],
                        "$FN-M": fn_info["FN_M"],
                        "$FN-E": fn_info["FN_E"],
                        "$FN-A": fn_info["FN_A"],
                    }
                    
                    records.append(record)
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(f"Error parsing MFT record: {e}")
        
        logger.info(f"MFT parsing complete - Extracted {len(records)} records")
        
        return pd.DataFrame(records)


class USNJournalParser:
    """
    Parser for Update Sequence Number Journal ($UsnJrnl:$J).
    
    Extracts:
    - File reference numbers (FRN)
    - Timestamps of change events
    - Reason codes for changes (e.g., DATA_EXTEND, BASIC_INFO_CHANGE)
    - File names and parent FRNs
    """
    
    # USN Reason flags
    REASON_FLAGS = {
        0x00000001: "DATA_OVERWRITE",
        0x00000002: "DATA_EXTEND",
        0x00000004: "DATA_TRUNCATION",
        0x00000010: "NAMED_DATA_OVERWRITE",
        0x00000020: "NAMED_DATA_EXTEND",
        0x00000040: "NAMED_DATA_TRUNCATION",
        0x00000100: "FILE_CREATE",
        0x00000200: "FILE_DELETE",
        0x00001000: "EA_CHANGE",
        0x00002000: "SECURITY_CHANGE",
        0x00004000: "RENAME_OLD_NAME",
        0x00008000: "RENAME_NEW_NAME",
        0x00010000: "INDEXABLE_CHANGE",
        0x00020000: "BASIC_INFO_CHANGE",
        0x00040000: "HARD_LINK_CHANGE",
        0x00080000: "COMPRESSION_CHANGE",
        0x00100000: "ENCRYPTION_CHANGE",
        0x00200000: "OBJECT_ID_CHANGE",
        0x00400000: "REPARSE_POINT_CHANGE",
        0x00800000: "STREAM_CHANGE",
        0x80000000: "CLOSE"
    }
    
    @staticmethod
    def usn_format_timestamp(dt_obj: Optional[datetime]) -> Optional[str]:
        """Format datetime object to string with microsecond precision."""
        if dt_obj is None:
            return None
        try:
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def usn_parse_reason_flags(reason_code: int) -> str:
        """Parse reason code integer into list of flag names."""
        if reason_code == 0:
            return "NONE"
        
        flags = []
        for mask, name in USNJournalParser.REASON_FLAGS.items():
            if reason_code & mask:
                flags.append(name)
        
        return " | ".join(flags) if flags else f"UNKNOWN_0x{reason_code:08X}"
    
    @staticmethod
    def usn_check_basic_detection_pattern(reason_code: int) -> bool:
        """Check if the reason code contains BASIC_INFO_CHANGE flag."""
        return bool(reason_code & 0x00020000)
    
    @staticmethod
    def usn_check_close_pattern(reason_code: int) -> bool:
        """Check if the reason code contains CLOSE flag."""
        return bool(reason_code & 0x80000000)
    
    @staticmethod
    def usn_check_file_create_pattern(reason_code: int) -> bool:
        """Check if the reason code contains FILE_CREATE flag."""
        return bool(reason_code & 0x00000100)
    
    @classmethod
    def parse_usnjrnl(cls, usnjrnl_path: Path) -> pd.DataFrame:
        """
        Parse the UsnJrnl file and extract all relevant metadata.
        
        Args:
            usnjrnl_path: Path to the $UsnJrnl:$J file
            
        Returns:
            pandas.DataFrame: Parsed UsnJrnl data with reason flags and detection patterns
        """
        records = []
        
        logger.info(f"Parsing UsnJrnl:J file: {usnjrnl_path.name}")
        
        with open(usnjrnl_path, "rb") as usn_file:
            parser = ChangeJournalParser(usn_file)
            
            error_count = 0
            
            for usn_record in parser.usn_records():
                try:

                    usn = usn_record.get_usn()
                    frn = usn_record.get_file_reference_number() & 0xFFFFFFFFFFFF
                    parent_frn = usn_record.get_parent_file_reference_number() & 0xFFFFFFFFFFFF
                    timestamp = cls.usn_format_timestamp(usn_record.get_timestamp())
                    filename = usn_record.get_file_name()
                    reason_code = usn_record.get_reason()
                    reason_flags = cls.usn_parse_reason_flags(reason_code)
                    source_info = usn_record.get_source_info()
                    
                    has_basic_info_change = cls.usn_check_basic_detection_pattern(reason_code)
                    has_close = cls.usn_check_close_pattern(reason_code)
                    has_file_create = cls.usn_check_file_create_pattern(reason_code)
                    
                    record = {
                        "USN": usn,
                        "FRN": frn,
                        "ParentFRN": parent_frn,
                        "Timestamp": timestamp,
                        "FileName": filename,
                        "ReasonCode": reason_code,
                        "ReasonFlags": reason_flags,
                        "SourceInfo": source_info,
                        "HasBasicInfoChange": has_basic_info_change,
                        "HasClose": has_close,
                        "HasFileCreate": has_file_create,
                    }
                    
                    records.append(record)
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(f"Error parsing USNJrnl:J record: {e}")
        
        logger.info(f"UsnJrnlL:J parsing complete - Extracted {len(records)} records")
        
        return pd.DataFrame(records)


class LogFileParser:
    """
    Parser for NTFS Transaction Log File ($LogFile).
    
    Extracts:
    - Transaction records and their operation types
    - Timestamp change events (UpdateResidentValue operations on timestamps)
    - Old vs. new timestamp values during metadata updates
    """
    
    # NTFS Operation Codes
    OPERATION_CODES = {
        0x00: "Noop",
        0x01: "CompensationLogRecord",
        0x02: "InitializeFileRecordSegment",
        0x03: "DeallocateFileRecordSegment",
        0x04: "WriteEndOfFileRecordSegment",
        0x05: "CreateAttribute",
        0x06: "DeleteAttribute",
        0x07: "UpdateResidentValue",
        0x08: "UpdateNonresidentValue",
        0x09: "UpdateMappingPairs",
        0x0A: "DeleteDirtyClusters",
        0x0B: "SetNewAttributeSizes",
        0x0C: "AddIndexEntryRoot",
        0x0D: "DeleteIndexEntryRoot",
        0x0E: "AddIndexEntryAllocation",
        0x0F: "DeleteIndexEntryAllocation",
        0x10: "WriteEndOfIndexBuffer",
        0x11: "SetIndexEntryVcnRoot",
        0x12: "SetIndexEntryVcnAllocation",
        0x13: "UpdateFileNameRoot",
        0x14: "UpdateFileNameAllocation",
        0x15: "SetBitsInNonresidentBitMap",
        0x16: "ClearBitsInNonresidentBitMap",
        0x17: "HotFix",
        0x18: "EndTopLevelAction",
        0x19: "PrepareTransaction",
        0x1A: "CommitTransaction",
        0x1B: "ForgetTransaction",
        0x1C: "OpenNonresidentAttribute",
        0x1D: "OpenAttributeTableDump",
        0x1E: "AttributeNamesDump",
        0x1F: "DirtyPageTableDump",
        0x20: "TransactionTableDump",
        0x21: "UpdateRecordDataRoot",
        0x22: "UpdateRecordDataAllocation",
        0x23: "UpdateRelativeDataIndex",
        0x24: "UpdateRelativeDataAllocation",
        0x25: "ZeroEndOfFileRecord"
    }
    
    @staticmethod
    def logfile_get_operation_name(op_code: int) -> str:
        """Get human-readable operation name."""
        return LogFileParser.OPERATION_CODES.get(op_code, f"Unknown_0x{op_code:02X}")
    
    @staticmethod
    def logfile_filetime_to_datetime(filetime_bytes: bytes) -> Optional[str]:
        """Convert FILETIME (8-byte little-endian) to datetime string."""
        if not filetime_bytes or len(filetime_bytes) != 8:
            return None
        
        try:
            filetime = struct.unpack('<Q', filetime_bytes)[0]
            EPOCH_DIFF = 116444736000000000
            
            if filetime == 0:
                return None
            
            microseconds = (filetime - EPOCH_DIFF) // 10
            dt = datetime(1970, 1, 1) + timedelta(microseconds=microseconds)
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        except (struct.error, ValueError, OSError, OverflowError):
            return None
    
    @staticmethod
    def logfile_extract_timestamps_from_buffer(data_buffer: bytes, attribute_offset: int) -> Dict[str, Optional[str]]:
        """
        Extract timestamps from redo/undo data buffer based on attribute offset.
        
        NTFS Standard Information attribute has 4 timestamps at specific offsets:
        - 0x18: All 4 timestamps (C, M, E, A)
        - 0x20: M, E, A timestamps
        - 0x28: E, A timestamps
        - 0x30: A timestamp
        """
        timestamps = {
            "C": None,
            "M": None,
            "E": None,
            "A": None
        }
        
        if not data_buffer or len(data_buffer) < 8:
            return timestamps
        
        try:
            if attribute_offset == 0x18:
                if len(data_buffer) >= 32:
                    timestamps["C"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[0:8])
                    timestamps["M"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[8:16])
                    timestamps["E"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[16:24])
                    timestamps["A"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[24:32])
            
            elif attribute_offset == 0x20:
                if len(data_buffer) >= 24:
                    timestamps["M"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[0:8])
                    timestamps["E"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[8:16])
                    timestamps["A"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[16:24])
            
            elif attribute_offset == 0x28:
                if len(data_buffer) >= 16:
                    timestamps["E"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[0:8])
                    timestamps["A"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[8:16])
            
            elif attribute_offset == 0x30:
                if len(data_buffer) >= 8:
                    timestamps["A"] = LogFileParser.logfile_filetime_to_datetime(data_buffer[0:8])
        
        except Exception as e:
            logger.debug(f"Error extracting timestamps from buffer: {e}")
        
        return timestamps
    
    @staticmethod
    def logfile_is_timestamp_change_record(record) -> bool:
        """Check if a LogFile record represents a timestamp change event."""
        try:
            if record.get_redo_operation() != 0x07:  # UpdateResidentValue
                return False
            
            record_offset = record.get_record_offset()
            if record_offset != 0x38:  # Offset for Standard Information in file record
                return False
            
            attr_offset = record.get_attribute_offset()
            if attr_offset < 0x18 or attr_offset > 0x30:  # Timestamp attribute ranges
                return False
            
            return True
        
        except Exception as e:
            logger.debug(f"Error checking timestamp change record: {e}")
            return False
    
    @classmethod
    def parse_logfile(cls, logfile_path: Path) -> pd.DataFrame:
        """
        Parse the LogFile and extract all relevant transaction records.
        
        Args:
            logfile_path: Path to the $LogFile
            
        Returns:
            pandas.DataFrame: Parsed LogFile data including timestamp change events
        """
        records = []
        
        logger.info(f"Parsing LogFile: {logfile_path.name}")
        
        with open(logfile_path, "rb") as logfile:
            parser = LogFileParser(logfile)
            parser.collect_lsns()
            
            timestamp_change_count = 0
            error_count = 0
            
            for record in parser.parse_ntfs_records():
                try:

                    lsn = record.get_lsn()
                    redo_op = record.get_redo_operation()
                    undo_op = record.get_undo_operation()
                    redo_op_name = cls.logfile_get_operation_name(redo_op)
                    undo_op_name = cls.logfile_get_operation_name(undo_op)
                    
                    try:
                        record_offset = record.get_record_offset()
                    except Exception:
                        record_offset = None
                
                    try:
                        attribute_offset = record.get_attribute_offset()
                    except Exception:
                        attribute_offset = None
                
                    try:
                        target_vcn = record.get_target_vcn()
                    except Exception:
                        target_vcn = None
                
                    try:
                        mft_target_number = record.calculate_mft_target_number()
                    except Exception:
                        mft_target_number = None
                
                    try:
                        redo_data = record.get_redo_data()
                    except Exception:
                        redo_data = None
                
                    try:
                        undo_data = record.get_undo_data()
                    except Exception:
                        undo_data = None
                    
                    is_timestamp_change = cls.logfile_is_timestamp_change_record(record)
                    
                    # Try to extract timestamp data from redo buffer
                    undo_timestamps = {"C": None, "M": None, "E": None, "A": None}
                    redo_timestamps = {"C": None, "M": None, "E": None, "A": None}
                
                    if is_timestamp_change and attribute_offset is not None:
                        timestamp_change_count += 1
                        if undo_data:
                            undo_timestamps = cls.logfile_extract_timestamps_from_buffer(undo_data, attribute_offset)
                        if redo_data:
                            redo_timestamps = cls.logfile_extract_timestamps_from_buffer(redo_data, attribute_offset)
                    
                    log_record = {
                        "LSN": lsn,
                        "RedoOP": redo_op,
                        "UndoOP": undo_op,
                        "RedoOPName": redo_op_name,
                        "UndoOPName": undo_op_name,
                        "RecordOffset": record_offset,
                        "AttributeOffset": attribute_offset,
                        "TargetVCN": target_vcn,
                        "TargetFRN": mft_target_number,
                        "IsTimestampChange": is_timestamp_change,
                        "Undo_$SI-C": undo_timestamps["C"],
                        "Undo_$SI-M": undo_timestamps["M"],
                        "Undo_$SI-E": undo_timestamps["E"],
                        "Undo_$SI-A": undo_timestamps["A"],
                        "Redo_$SI-C": redo_timestamps["C"],
                        "Redo_$SI-M": redo_timestamps["M"],
                        "Redo_$SI-E": redo_timestamps["E"],
                        "Redo_$SI-A": redo_timestamps["A"]
                    }
                    
                    records.append(log_record)
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(f"Error parsing LogFile record: {e}")
        
        logger.info(f"LogFile parsing complete - Extracted {len(records)} records")
        
        return pd.DataFrame(records)


class RawFilesParser:
    """
    Orchestrator class for parsing all NTFS system files.
    
    Coordinates parsing of $MFT, $LogFile, and $UsnJrnl:$J, generating
    three individual CSV output files for downstream processing.
    """
    
    def __init__(self, logger_obj=None):
        """
        Initialize the parser.
        
        Args:
            logger_obj: Optional logger object. If not provided, uses module logger.
        """
        self.logger = logger_obj if logger_obj else logger
    
    def parse_all(self, exported_files_dir: Path, output_dir: Path) -> Dict[str, Tuple[bool, str, Optional[pd.DataFrame]]]:
        """
        Parse all three NTFS system files from the exported files directory.
        
        Args:
            exported_files_dir: Directory containing exported $MFT, $LogFile, and $UsnJrnl:$J files
            output_dir: Output directory for parsed CSV files (created by ntfs_timestomping_detector.py)
            
        Returns:
            dict: Results for each parser with format:
                {
                    'mft': (success: bool, message: str, dataframe: DataFrame or None),
                    'logfile': (success: bool, message: str, dataframe: DataFrame or None),
                    'usnjrnl': (success: bool, message: str, dataframe: DataFrame or None)
                }
        """
        exported_files_dir = Path(exported_files_dir)
        output_dir = Path(output_dir)
        
        # Define expected file paths
        mft_path = exported_files_dir / "$MFT"
        logfile_path = exported_files_dir / "$LogFile"
        usnjrnl_path = exported_files_dir / "$UsnJrnl_$J"
        
        #Verify input files exist
        missing_files = []
        for file_path in [mft_path, logfile_path, usnjrnl_path]:
            if not file_path.is_file():
                missing_files.append(file_path.name)

        if missing_files:
            self.logger.error(f"Missing input files: {', '.join(missing_files)}")
            return {
                'mft': (False, f"Missing input files: {', '.join(missing_files)}", None),
                'logfile': (False, f"Missing input files: {', '.join(missing_files)}", None),
                'usnjrnl': (False, f"Missing input files: {', '.join(missing_files)}", None)
            }

        results = {}
        
        # Parse MFT
        self.logger.info("Parsing $MFT...")
        try:
            df_mft = MFTParser.parse_mft(mft_path)
            output_file = output_dir / "MFT_parsed.csv"
            df_mft.to_csv(output_file, index=False, encoding="utf-8")
            self.logger.info(f"MFT saved to: {output_file}")
            results['mft'] = (True, f"Successfully parsed {len(df_mft)} MFT records", df_mft)
        except Exception as e:
            msg = f"Error parsing MFT: {str(e)}"
            self.logger.error(msg)
            results['mft'] = (False, msg, None)
        
        # Parse LogFile
        self.logger.info("Parsing $LogFile...")
        try:
            df_logfile = LogFileParser.parse_logfile(logfile_path)
            output_file = output_dir / "LogFile_parsed.csv"
            df_logfile.to_csv(output_file, index=False, encoding="utf-8")
            self.logger.info(f"LogFile saved to: {output_file}")
            results['logfile'] = (True, f"Successfully parsed {len(df_logfile)} LogFile records", df_logfile)
        except Exception as e:
            msg = f"Error parsing LogFile: {str(e)}"
            self.logger.error(msg)
            results['logfile'] = (False, msg, None)
        
        # Parse UsnJrnl
        self.logger.info("Parsing $UsnJrnl:$J...")
        try:
            df_usnjrnl = USNJournalParser.parse_usnjrnl(usnjrnl_path)
            output_file = output_dir / "UsnJrnl:$J_parsed.csv"
            df_usnjrnl.to_csv(output_file, index=False, encoding="utf-8")
            self.logger.info(f"UsnJrnl saved to: {output_file}")
            results['usnjrnl'] = (True, f"Successfully parsed {len(df_usnjrnl)} UsnJrnl records", df_usnjrnl)
        except Exception as e:
            msg = f"Error parsing UsnJrnl: {str(e)}"
            self.logger.error(msg)
            results['usnjrnl'] = (False, msg, None)
        
        self.logger.info("Parsing complete")
        
        return results