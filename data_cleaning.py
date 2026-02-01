"""
Generic Data Cleaning Module
Handles data loading and preprocessing for any JSON/JSONL review dataset
"""

import json
import re
import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, List
from datetime import timedelta


class TimingTracker:
    """Utility class to track execution times"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.total_start_time = None
    
    def start_total(self):
        self.total_start_time = time.time()
    
    def start(self, step_name: str):
        self.start_times[step_name] = time.time()
    
    def stop(self, step_name: str):
        if step_name in self.start_times:
            elapsed = time.time() - self.start_times[step_name]
            self.timings[step_name] = elapsed
            del self.start_times[step_name]
            return elapsed
        return None
    
    def get_total_time(self) -> float:
        if self.total_start_time:
            return time.time() - self.total_start_time
        return 0.0
    
    def format_time(self, seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            return str(timedelta(seconds=int(seconds)))
    
    def print_report(self):
        print("\n" + "=" * 60)
        print("TIMING REPORT")
        print("=" * 60)
        
        if not self.timings:
            print("No timing data available")
            return
        
        total_time = self.get_total_time()
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Step':<45} {'Time':>12}")
        print("-" * 60)
        
        for step_name, duration in sorted_timings:
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{step_name:<45} {self.format_time(duration):>8} ({percentage:5.1f}%)")
        
        print("-" * 60)
        print(f"{'TOTAL EXECUTION TIME':<45} {self.format_time(total_time):>12}")
        print("=" * 60)


class GenericDataCleaner:
    """Generic data cleaning class for any review dataset"""
    
    def __init__(self, 
                 text_column: str,
                 label_column: str,
                 additional_features: List[str],
                 cleaning_options: Dict[str, bool],
                 verbose: bool = True):
        """
        Initialize data cleaner
        
        Args:
            text_column: Name of the text column
            label_column: Name of the label column
            additional_features: List of additional feature column names
            cleaning_options: Dict of cleaning operations to perform
            verbose: Whether to print detailed statistics
        """
        self.text_column = text_column
        self.label_column = label_column
        self.additional_features = additional_features
        self.cleaning_options = cleaning_options
        self.verbose = verbose
        self.timer = TimingTracker()
        self.stats = {}
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from JSON or JSONL file"""
        self.timer.start("Data Loading")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 1: DATA LOADING")
            print("=" * 60)
            print(f"Loading data from: {data_path}")
        
        # Determine file format
        is_jsonl = data_path.endswith('.jsonl')
        
        try:
            if is_jsonl:
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            data.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            if self.verbose:
                                print(f"Warning: Skipping malformed line {i+1}: {e}")
                df = pd.DataFrame(data)
            else:
                # Regular JSON file
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both list of objects and single object with nested data
                    if isinstance(data, dict):
                        # Try common keys for nested data
                        for key in ['data', 'reviews', 'items', 'records']:
                            if key in data and isinstance(data[key], list):
                                data = data[key]
                                break
                df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Error loading data file: {e}")
        
        self.timer.stop("Data Loading")
        
        if self.verbose:
            print(f"✓ Data loaded successfully")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        
        # Validate required columns exist
        missing_cols = []
        if self.text_column not in df.columns:
            missing_cols.append(self.text_column)
        if self.label_column not in df.columns:
            missing_cols.append(self.label_column)
        
        if missing_cols:
            raise ValueError(
                f"Required columns not found in dataset: {missing_cols}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Please check your --text-column and --label-column arguments"
            )
        
        self.stats['initial_rows'] = len(df)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled cleaning operations"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 2: DATA CLEANING")
            print("=" * 60)
        
        self.timer.start_total()
        operations_performed = []
        
        if self.cleaning_options.get('duplicates', False):
            df = self._remove_duplicates(df)
            operations_performed.append('duplicates')
        
        if self.cleaning_options.get('missing_values', False):
            df = self._handle_missing_values(df)
            operations_performed.append('missing_values')
        
        if self.cleaning_options.get('remove_empty_text', False):
            df = self._remove_empty_text(df)
            operations_performed.append('remove_empty_text')
        
        if self.cleaning_options.get('type_errors', False):
            df = self._fix_type_errors(df)
            operations_performed.append('type_errors')
        
        if self.cleaning_options.get('text_garbling', False):
            df = self._fix_text_garbling(df)
            operations_performed.append('text_garbling')
        
        if self.verbose:
            print(f"\n✓ Cleaning completed")
            if operations_performed:
                print(f"  Operations performed: {', '.join(operations_performed)}")
            else:
                print(f"  No cleaning operations performed")
            print(f"  Final shape: {df.shape}")
            self.timer.print_report()
        
        self.stats['final_rows'] = len(df)
        self.stats['rows_removed'] = self.stats['initial_rows'] - self.stats['final_rows']
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows, handling unhashable types"""
        self.timer.start("Remove Duplicates")
        
        def make_hashable(val):
            """Convert unhashable types to hashable equivalents"""
            if isinstance(val, list):
                return tuple(make_hashable(item) for item in val)
            elif isinstance(val, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
            elif isinstance(val, set):
                return frozenset(make_hashable(item) for item in val)
            else:
                return val
        
        initial_rows = len(df)
        
        # Create a temporary dataframe with hashable values
        df_temp = df.copy()
        for col in df_temp.columns:
            # Check if column contains unhashable types
            try:
                df_temp[col].apply(hash)
            except TypeError:
                # Convert unhashable types to hashable equivalents
                df_temp[col] = df_temp[col].apply(make_hashable)
        
        # Find duplicates using the hashable version
        duplicates_mask = df_temp.duplicated(keep='first')
        
        # Apply the mask to the original dataframe
        df = df[~duplicates_mask].copy()
        
        duplicates_removed = initial_rows - len(df)
        
        self.timer.stop("Remove Duplicates")
        
        if self.verbose and duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_rows*100:.2f}%)")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in text and label columns"""
        self.timer.start("Handle Missing Values")
        
        initial_rows = len(df)
        
        # Remove rows with missing text or labels
        df = df.dropna(subset=[self.text_column, self.label_column])
        
        rows_removed = initial_rows - len(df)
        
        self.timer.stop("Handle Missing Values")
        
        if self.verbose and rows_removed > 0:
            print(f"✓ Removed {rows_removed} rows with missing values ({rows_removed/initial_rows*100:.2f}%)")
        
        return df
    
    def _remove_empty_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with empty or whitespace-only text"""
        self.timer.start("Remove Empty Text")
        
        initial_rows = len(df)
        
        # Convert to string and strip whitespace
        df[self.text_column] = df[self.text_column].astype(str).str.strip()
        
        # Remove empty strings
        df = df[df[self.text_column].str.len() > 0]
        
        rows_removed = initial_rows - len(df)
        
        self.timer.stop("Remove Empty Text")
        
        if self.verbose and rows_removed > 0:
            print(f"✓ Removed {rows_removed} rows with empty text ({rows_removed/initial_rows*100:.2f}%)")
        
        return df
    
    def _fix_type_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix type errors in label and numeric columns"""
        self.timer.start("Fix Type Errors")
        
        initial_rows = len(df)
        
        # Handle label column
        if self.label_column in df.columns:
            # First, handle list/array types - extract first element
            def extract_scalar(val):
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    return val[0]
                elif isinstance(val, (list, tuple)) and len(val) == 0:
                    return None
                else:
                    return val
            
            df[self.label_column] = df[self.label_column].apply(extract_scalar)
            
            # Then convert to numeric
            if df[self.label_column].dtype == object:
                df[self.label_column] = pd.to_numeric(df[self.label_column], errors='coerce')
                
                # Remove rows where conversion failed (NaN)
                rows_before = len(df)
                df = df.dropna(subset=[self.label_column])
                rows_removed = rows_before - len(df)
                
                if self.verbose and rows_removed > 0:
                    print(f"  Removed {rows_removed} rows with invalid label values")
            
            # Ensure labels are proper numeric type (not object)
            if df[self.label_column].dtype == 'float64':
                # Check if all values are integers
                if (df[self.label_column] == df[self.label_column].astype(int)).all():
                    df[self.label_column] = df[self.label_column].astype(int)
        
        # Handle additional numeric features
        for col in self.additional_features:
            if col in df.columns:
                # Handle list/array types
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x)
                
                if df[col].dtype == object:
                    # Try numeric conversion
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        self.timer.stop("Fix Type Errors")
        
        if self.verbose:
            print(f"✓ Fixed type errors in label and numeric columns")
            print(f"  Label column type: {df[self.label_column].dtype}")
        
        return df
    
    def _fix_text_garbling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix text garbling issues"""
        self.timer.start("Fix Text Garbling")
        
        # Ensure text column is string type
        df[self.text_column] = df[self.text_column].astype(str)
        
        # Remove leading/trailing whitespace
        df[self.text_column] = df[self.text_column].str.strip()
        
        # Remove control characters
        df[self.text_column] = df[self.text_column].apply(
            lambda x: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', x) if pd.notna(x) else x
        )
        
        # Fix multiple spaces
        df[self.text_column] = df[self.text_column].str.replace(r'\s+', ' ', regex=True)
        
        self.timer.stop("Fix Text Garbling")
        
        if self.verbose:
            print(f"✓ Fixed text garbling in text column")
        
        return df


def load_and_clean_data(data_path: str,
                       text_column: str,
                       label_column: str,
                       additional_features: List[str] = None,
                       cleaning_options: Dict[str, bool] = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Main function to load and clean data
    
    Args:
        data_path: Path to the data file
        text_column: Name of text feature column
        label_column: Name of label column
        additional_features: List of additional feature columns
        cleaning_options: Dict of cleaning operations to perform
        verbose: Whether to print detailed statistics
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if additional_features is None:
        additional_features = []
    
    if cleaning_options is None:
        cleaning_options = {
            'missing_values': True,
            'duplicates': True,
            'text_garbling': True,
            'type_errors': True,
            'remove_empty_text': True,
        }
    
    cleaner = GenericDataCleaner(
        text_column=text_column,
        label_column=label_column,
        additional_features=additional_features,
        cleaning_options=cleaning_options,
        verbose=verbose
    )
    
    df = cleaner.load_data(data_path)
    df = cleaner.clean_data(df)
    
    return df


def load_data_without_cleaning(data_path: str,
                               text_column: str,
                               label_column: str,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Load data without any cleaning operations
    
    Args:
        data_path: Path to the data file
        text_column: Name of text feature column
        label_column: Name of label column
        verbose: Whether to print statistics
    
    Returns:
        pd.DataFrame: Raw dataframe
    """
    if verbose:
        print("\n" + "=" * 60)
        print("LOADING DATA (NO CLEANING)")
        print("=" * 60)
    
    # Determine file format
    is_jsonl = data_path.endswith('.jsonl')
    
    try:
        if is_jsonl:
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            df = pd.DataFrame(data)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for key in ['data', 'reviews', 'items', 'records']:
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            break
            df = pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"Error loading data file: {e}")
    
    if verbose:
        print(f"✓ Data loaded")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    
    # Validate columns
    missing_cols = []
    if text_column not in df.columns:
        missing_cols.append(text_column)
    if label_column not in df.columns:
        missing_cols.append(label_column)
    
    if missing_cols:
        raise ValueError(
            f"Required columns not found: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    return df
