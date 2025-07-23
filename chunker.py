#!/usr/bin/env python3
"""
Markdown Paragraph Chunker
==========================

This module provides functionality to read markdown files and identify paragraphs,
creating a structured representation of the document content.

Features:
- Ignores figures with pattern: ![](_page_XXX_Figure_X.jpeg)
- Supports various markdown elements (headers, tables, lists, etc.)
- Creates sliding windows for chunking
- Uses LLM for semantic chunking
"""

import os
import re
from typing import List, Dict, Any
import dspy
from datetime import datetime
import json
import sys
filename = sys.argv[1].replace('\\', '').replace(' ', '')
print(filename)



def is_ignored_figure(line: str) -> bool:
    """
    Check if a line represents a figure that should be ignored.
    
    Args:
        line (str): The line to check
        
    Returns:
        bool: True if the line should be ignored as a figure
    """
    # Check for the specific pattern: ![](_page_XXX_Figure_X.jpeg)
    if line.startswith('![') and '](' in line:
        # Extract the URL part
        url_start = line.find('](') + 2
        url_end = line.find(')', url_start)
        if url_end != -1:
            url = line[url_start:url_end]
            # Check if it matches the pattern: _page_XXX_Figure_X.jpeg
            if re.match(r'^_page_\d+_Figure_\d+\.(jpeg|jpg|png|gif|svg)$', url):
                return True
    return False


def read_markdown_paragraphs(markdown_file_path: str) -> List[Dict[str, str]]:
    """
    Read a markdown file and identify paragraphs, returning a list of dictionaries.
    
    Args:
        markdown_file_path (str): Path to the markdown file to read
        
    Returns:
        List[Dict[str, str]]: List of dictionaries, each containing a paragraph
                             with key 'md-text' and paragraph content as value
                             
    Raises:
        FileNotFoundError: If the markdown file doesn't exist
        Exception: For other file reading errors
    """
    
    if not os.path.exists(markdown_file_path):
        raise FileNotFoundError(f"Markdown file not found: {markdown_file_path}")
    
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        raise Exception(f"Error reading markdown file: {e}")
    
    # Split content into lines
    lines = content.split('\n')
    
    paragraphs = []
    current_paragraph = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines (they mark paragraph boundaries)
        if not line:
            if current_paragraph:
                # Join the current paragraph and add to results
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:  # Only add non-empty paragraphs
                    paragraphs.append({'md-text': paragraph_text})
                current_paragraph = []
            continue
        
        # Skip ignored figures
        if is_ignored_figure(line):
            continue
        
        # Check if this line is a markdown element that should start a new paragraph
        is_new_paragraph_element = (
            line.startswith('#') or  # Headers
            line.startswith('|') or  # Table rows
            line.startswith('![') or  # Images
            line.startswith('```') or  # Code blocks
            line.startswith('>') or  # Blockquotes
            line.startswith('- ') or  # List items
            line.startswith('* ') or  # List items
            line.startswith('+ ') or  # List items
            line.startswith('1. ') or  # Numbered list items
            re.match(r'^\d+\.\s', line) or  # Numbered list items with regex
            line.startswith('---') or  # Horizontal rules
            line.startswith('***') or  # Horizontal rules
            line.startswith('___') or  # Horizontal rules
            line.startswith('[') and '](' in line or  # Links
            re.match(r'^\s*\$\$.*\$\$\s*$', line)  # Only pure $$...$$ block is LaTeX
        )
        
        if is_new_paragraph_element and current_paragraph:
            # Save current paragraph and start a new one
            paragraph_text = ' '.join(current_paragraph).strip()
            if paragraph_text:
                paragraphs.append({'md-text': paragraph_text})
            current_paragraph = [line]
        else:
            # Add line to current paragraph
            current_paragraph.append(line)
    
    # Don't forget the last paragraph
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph).strip()
        if paragraph_text:
            paragraphs.append({'md-text': paragraph_text})
    
    return paragraphs


def read_markdown_paragraphs_advanced(markdown_file_path: str, 
                                    preserve_formatting: bool = True,
                                    include_metadata: bool = False) -> List[Dict[str, Any]]:
    """
    Advanced markdown paragraph reader with additional options.
    
    Args:
        markdown_file_path (str): Path to the markdown file to read
        preserve_formatting (bool): Whether to preserve markdown formatting in paragraphs
        include_metadata (bool): Whether to include paragraph metadata (type, level, etc.)
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries with paragraph content and optional metadata
    """
    
    if not os.path.exists(markdown_file_path):
        raise FileNotFoundError(f"Markdown file not found: {markdown_file_path}")
    
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        raise Exception(f"Error reading markdown file: {e}")
    
    lines = content.split('\n')
    paragraphs = []
    current_paragraph = []
    paragraph_type = "text"
    paragraph_level = 0
    
    for i, line in enumerate(lines):
        original_line = line
        line = line.strip()
        
        # Determine line type
        if line.startswith('#'):
            # Header
            level = len(line) - len(line.lstrip('#'))
            if current_paragraph:
                # Save current paragraph
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:
                    para_dict = {'md-text': paragraph_text}
                    if include_metadata:
                        para_dict.update({
                            'type': paragraph_type,
                            'level': paragraph_level,
                            'line_number': i - len(current_paragraph)
                        })
                    paragraphs.append(para_dict)
            
            # Start new header paragraph
            current_paragraph = [line]
            paragraph_type = "header"
            paragraph_level = level
            
        elif line.startswith('|'):
            # Table row
            if current_paragraph and paragraph_type != "table":
                # Save current paragraph
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:
                    para_dict = {'md-text': paragraph_text}
                    if include_metadata:
                        para_dict.update({
                            'type': paragraph_type,
                            'level': paragraph_level,
                            'line_number': i - len(current_paragraph)
                        })
                    paragraphs.append(para_dict)
            
            current_paragraph = [line]
            paragraph_type = "table"
            paragraph_level = 0
            
        elif line.startswith('![') or line.startswith('```') or line.startswith('>'):
            # Skip ignored figures
            if line.startswith('![') and is_ignored_figure(line):
                continue
                
            # Image, code block, or blockquote
            if current_paragraph:
                # Save current paragraph
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:
                    para_dict = {'md-text': paragraph_text}
                    if include_metadata:
                        para_dict.update({
                            'type': paragraph_type,
                            'level': paragraph_level,
                            'line_number': i - len(current_paragraph)
                        })
                    paragraphs.append(para_dict)
            
            current_paragraph = [line]
            if line.startswith('!['):
                paragraph_type = "image"
            elif line.startswith('```'):
                paragraph_type = "code"
            else:
                paragraph_type = "quote"
            paragraph_level = 0
            
        elif (line.startswith('- ') or line.startswith('* ') or 
              line.startswith('+ ') or re.match(r'^\d+\.\s', line)):
            # List item
            if current_paragraph and paragraph_type != "list":
                # Save current paragraph
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:
                    para_dict = {'md-text': paragraph_text}
                    if include_metadata:
                        para_dict.update({
                            'type': paragraph_type,
                            'level': paragraph_level,
                            'line_number': i - len(current_paragraph)
                        })
                    paragraphs.append(para_dict)
            
            current_paragraph = [line]
            paragraph_type = "list"
            paragraph_level = 0
            
        elif re.match(r'^\s*\$\$.*\$\$\s*$', line):  # Only pure $$...$$ block is LaTeX
            # LaTeX content
            if current_paragraph and paragraph_type != "latex":
                # Save current paragraph
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:
                    para_dict = {'md-text': paragraph_text}
                    if include_metadata:
                        para_dict.update({
                            'type': paragraph_type,
                            'level': paragraph_level,
                            'line_number': i - len(current_paragraph)
                        })
                    paragraphs.append(para_dict)
            current_paragraph = [line]
            paragraph_type = "latex"
            paragraph_level = 0
            
        elif not line:
            # Empty line - paragraph boundary
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph).strip()
                if paragraph_text:
                    para_dict = {'md-text': paragraph_text}
                    if include_metadata:
                        para_dict.update({
                            'type': paragraph_type,
                            'level': paragraph_level,
                            'line_number': i - len(current_paragraph)
                        })
                    paragraphs.append(para_dict)
                current_paragraph = []
                paragraph_type = "text"
                paragraph_level = 0
        else:
            # Regular text line
            if not preserve_formatting and paragraph_type == "text":
                # Remove markdown formatting for plain text
                line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)  # Bold
                line = re.sub(r'\*(.*?)\*', r'\1', line)      # Italic
                line = re.sub(r'`(.*?)`', r'\1', line)        # Inline code
                line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)  # Links
            
            current_paragraph.append(line)
    
    # Don't forget the last paragraph
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph).strip()
        if paragraph_text:
            para_dict = {'md-text': paragraph_text}
            if include_metadata:
                para_dict.update({
                    'type': paragraph_type,
                    'level': paragraph_level,
                    'line_number': len(lines) - len(current_paragraph)
                })
            paragraphs.append(para_dict)
    
    return paragraphs


def print_paragraphs(paragraphs: List[Dict[str, Any]], show_metadata: bool = False):
    """
    Print paragraphs in a readable format.
    
    Args:
        paragraphs (List[Dict[str, Any]]): List of paragraph dictionaries
        show_metadata (bool): Whether to show metadata if available
    """
    print(f"Found {len(paragraphs)} paragraphs:\n")
    
    for i, para in enumerate(paragraphs, 1):
        print(f"Paragraph {i}:")
        print(f"  Content: {para['md-text'][:100]}{'...' if len(para['md-text']) > 100 else ''}")
        
        if show_metadata and len(para) > 1:
            metadata = {k: v for k, v in para.items() if k != 'md-text'}
            print(f"  Metadata: {metadata}")
        print()

def create_sliding_windows_fifo_with_overlap(paragraphs: List[Dict[str, Any]], 
                                            window_size: int = 5,
                                            step_size: int = 1) -> List[List[Dict[str, Any]]]:
    """
    Create sliding windows with FIFO behavior and configurable step size.
    Automatically detects paragraph types if not present in metadata.
    
    Args:
        paragraphs (List[Dict[str, Any]]): List of paragraph dictionaries
        window_size (int): Size of each window (default: 5)
        step_size (int): How many paragraphs to shift between windows (default: 1 for true FIFO)
        
    Returns:
        List[List[Dict[str, Any]]]: List of window groups with FIFO behavior and type metadata
    """
    
    if not paragraphs:
        return []
    
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    if step_size <= 0:
        raise ValueError("Step size must be positive")
    
    if step_size >= window_size:
        raise ValueError("Step size must be less than window size for FIFO behavior")
    
    if window_size > len(paragraphs):
        # If window size is larger than available paragraphs, return all paragraphs as one window
        return [paragraphs]
    
    # First, enhance paragraphs with type detection if needed
    enhanced_paragraphs = []
    for para in paragraphs:
        enhanced_para = para.copy()
        
        # If type is missing or 'unknown', detect it from content
        if 'type' not in enhanced_para or enhanced_para.get('type') == 'unknown':
            enhanced_para['type'] = detect_paragraph_type_from_content(para)
        
        # Add level information for headers
        if enhanced_para['type'] == 'header':
            content = enhanced_para.get('md-text', '')
            level = len(content) - len(content.lstrip('#'))
            enhanced_para['level'] = level
        
        # Add LaTeX information for LaTeX paragraphs
        if enhanced_para['type'] == 'latex':
            content = enhanced_para.get('md-text', '')
            latex_info = extract_latex_info(content)
            enhanced_para['latex_info'] = latex_info
        
        enhanced_paragraphs.append(enhanced_para)
    
    windows = []
    
    # Create sliding windows with FIFO behavior
    # Each window shifts by step_size paragraphs
    for i in range(0, len(enhanced_paragraphs) - window_size + 1, step_size):
        window = enhanced_paragraphs[i:i + window_size]
        windows.append(window)
    
    return windows


def create_latex_aware_chunks(paragraphs: List[Dict[str, Any]], 
                             window_size: int = 5,
                             step_size: int = 1) -> List[List[Dict[str, Any]]]:
    """
    Create LaTeX-aware sliding windows that group related LaTeX content together.
    
    Args:
        paragraphs (List[Dict[str, Any]]): List of paragraph dictionaries
        window_size (int): Size of each window (default: 5)
        step_size (int): How many paragraphs to shift between windows (default: 1)
        
    Returns:
        List[List[Dict[str, Any]]]: List of window groups with LaTeX-aware grouping
    """
    
    if not paragraphs:
        return []
    
    # First, enhance paragraphs with LaTeX information
    enhanced_paragraphs = []
    for para in paragraphs:
        enhanced_para = para.copy()
        
        # Detect type if not present
        if 'type' not in enhanced_para:
            enhanced_para['type'] = detect_paragraph_type_from_content(para)
        
        # Add LaTeX information
        if enhanced_para['type'] == 'latex':
            content = enhanced_para.get('md-text', '')
            latex_info = extract_latex_info(content)
            enhanced_para['latex_info'] = latex_info
        
        enhanced_paragraphs.append(enhanced_para)
    
    # Create LaTeX-aware windows
    windows = []
    current_window = []
    
    for i, para in enumerate(enhanced_paragraphs):
        current_window.append(para)
        
        # Check if we should create a new window
        should_split = False
        
        # Split if window is full
        if len(current_window) >= window_size:
            should_split = True
        
        # Split if we encounter a major header (L1 or L2) and have content
        elif (para.get('type') == 'header' and 
              para.get('level', 0) <= 2 and 
              len(current_window) > 1):
            should_split = True
        
        # Split if we have a display math block and it's not the first item
        elif (para.get('type') == 'latex' and 
              para.get('latex_info', {}).get('latex_type') == 'display' and
              len(current_window) > 1):
            should_split = True
        
        if should_split:
            # Remove the current paragraph from this window and add to next
            current_window.pop()
            if current_window:
                windows.append(current_window)
            current_window = [para]
    
    # Add the last window if it has content
    if current_window:
        windows.append(current_window)
    
    return windows


def extract_latex_info(content: str) -> Dict[str, Any]:
    """
    Extract LaTeX-specific information from content.
    
    Args:
        content (str): Content that may contain LaTeX
        
    Returns:
        Dict[str, Any]: Dictionary with LaTeX analysis
    """
    latex_info = {
        'has_latex': False,
        'latex_type': None,
        'environments': [],
        'inline_math': [],
        'display_math': [],
        'commands': []
    }
    
    # Check for inline math with $
    inline_matches = re.findall(r'\$([^$]+)\$', content)
    if inline_matches:
        latex_info['has_latex'] = True
        latex_info['latex_type'] = 'inline'
        latex_info['inline_math'] = inline_matches
    
    # Check for display math with $$
    display_matches = re.findall(r'\$\$([^$]+)\$\$', content)
    if display_matches:
        latex_info['has_latex'] = True
        latex_info['latex_type'] = 'display'
        latex_info['display_math'] = display_matches
    
    # Check for LaTeX environments
    begin_matches = re.findall(r'\\begin\{([^}]+)\}', content)
    end_matches = re.findall(r'\\end\{([^}]+)\}', content)
    if begin_matches or end_matches:
        latex_info['has_latex'] = True
        latex_info['latex_type'] = 'environment'
        latex_info['environments'] = list(set(begin_matches + end_matches))
    
    # Check for LaTeX commands
    command_matches = re.findall(r'\\([a-zA-Z]+)\{', content)
    if command_matches:
        latex_info['has_latex'] = True
        if not latex_info['latex_type']:
            latex_info['latex_type'] = 'command'
        latex_info['commands'] = list(set(command_matches))
    
    # Check for display math with \[ \]
    bracket_matches = re.findall(r'(?<!\\)\\\[([^\]]+)\\\]', content)
    if bracket_matches:
        latex_info['has_latex'] = True
        latex_info['latex_type'] = 'display'
        latex_info['display_math'].extend(bracket_matches)
    
    # Check for inline math with \( \)
    paren_matches = re.findall(r'(?<!\\)\\\(([^)]+)\\\)', content)
    if paren_matches:
        latex_info['has_latex'] = True
        latex_info['latex_type'] = 'inline'
        latex_info['inline_math'].extend(paren_matches)
    
    return latex_info


def detect_paragraph_type_from_content(paragraph: Dict[str, Any]) -> str:
    """
    Detect paragraph type from content when metadata is not available.
    
    Args:
        paragraph (Dict[str, Any]): Paragraph dictionary
        
    Returns:
        str: Detected paragraph type
    """
    content = paragraph.get('md-text', '')
    
    if not content:
        return 'empty'
    
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # Check for headers
    if content.startswith('#'):
        return 'header'
    
    # Check for table rows
    if content.startswith('|') and '|' in content[1:]:
        return 'table'
    
    # Check for images (but skip ignored figures)
    if content.startswith('![') and '](' in content:
        if is_ignored_figure(content):
            return 'ignored_figure'
        return 'image'
    
    # Check for code blocks
    if content.startswith('```'):
        return 'code'
    
    # Check for blockquotes
    if content.startswith('>'):
        return 'quote'
    
    # Check for list items
    if (content.startswith('- ') or 
        content.startswith('* ') or 
        content.startswith('+ ') or
        re.match(r'^\d+\.\s', content)):
        return 'list'
    
    # Check for horizontal rules
    if content in ['---', '***', '___']:
        return 'rule'
    
    # Check for links
    if content.startswith('[') and '](' in content:
        return 'link'
    
    # Check for LaTeX content (only pure $$...$$ block)
    if re.match(r'^\s*\$\$.*\$\$\s*$', content):
        return 'latex'
    
    # Default to text
    return 'text'


def print_fifo_windows(windows: List[List[Dict[str, Any]]], 
                      show_metadata: bool = True,
                      max_chars: int = 80):
    """
    Print sliding windows highlighting FIFO behavior with proper type detection.
    
    Args:
        windows (List[List[Dict[str, Any]]]): List of window groups
        show_metadata (bool): Whether to show paragraph metadata
        max_chars (int): Maximum characters to show per paragraph
    """
    
    print(f"Created {len(windows)} sliding windows with FIFO behavior:\n")
    
    for i, window in enumerate(windows, 1):
        print(f"Window {i}:")
        
        for j, para in enumerate(window, 1):
            content = para.get('md-text', '')
            if show_metadata:
                para_type = para.get('type', 'unknown')
                level = para.get('level', 0)
                
                # Special handling for LaTeX paragraphs
                if para_type == 'latex':
                    latex_info = para.get('latex_info', {})
                    latex_type = latex_info.get('latex_type', 'unknown')
                    if latex_info.get('has_latex'):
                        print(f"  P{j}: ({para_type}:{latex_type}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
                    else:
                        print(f"  P{j}: ({para_type}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
                elif para_type == 'header' and level > 0:
                    print(f"  P{j}: ({para_type} L{level}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
                else:
                    print(f"  P{j}: ({para_type}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
            else:
                print(f"  P{j}: {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
        
        # Show FIFO behavior
        if i > 1:
            prev_window = windows[i-2]
            removed = prev_window[0]  # First paragraph from previous window
            added = window[-1]        # Last paragraph from current window
            print(f"  FIFO: '{removed.get('md-text', '')[:30]}...' → OUT, '{added.get('md-text', '')[:30]}...' → IN")
        
        print()


# Also update the main processing function to use the enhanced version
def process_paragraphs_fifo_for_llm(paragraphs: List[Dict[str, Any]], 
                                   window_size: int = 5,
                                   step_size: int = 1,
                                   format_type: str = "text",
                                   include_metadata: bool = True) -> tuple[List[List[Dict[str, Any]]], List[str]]:
    """
    Complete pipeline to process paragraphs with FIFO sliding windows for LLM analysis.
    Automatically detects paragraph types if not present in metadata.
    
    Args:
        paragraphs (List[Dict[str, Any]]): List of paragraph dictionaries
        window_size (int): Size of each window
        step_size (int): How many paragraphs to shift between windows (1 for true FIFO)
        format_type (str): Output format for LLM
        include_metadata (bool): Whether to include metadata in formatted text
        
    Returns:
        tuple: (raw_windows, formatted_windows)
            - raw_windows: List of window groups as dictionaries
            - formatted_windows: List of formatted text strings for LLM
    """
    
    # Create sliding windows with FIFO behavior and type detection
    raw_windows = create_sliding_windows_fifo_with_overlap(paragraphs, window_size, step_size)
        
    return raw_windows

def llm_for_context_chunking(windows: List[List[Dict[str, Any]]], 
                            output_filename: str = None) -> List[Dict[str, Any]]:
    """
    Use LLM to analyze windows and determine chunk boundaries based on semantic context.
    Adds chunk numbers, titles, and descriptions to each paragraph based on topic coherence.
    Writes to JSON file every 20 chunks if output_filename is provided.
    
    Args:
        windows (List[List[Dict[str, Any]]]): List of window groups from sliding window
        output_filename (str): Optional filename to write JSON output every 20 chunks
        
    Returns:
        List[Dict[str, Any]]: Enhanced paragraphs with chunk numbers, titles, and descriptions
    """
    
    # Configure DSPy to use local Ollama
    try:
        # Create LLM configuration for Ollama
        llm = dspy.LM(
            model="ollama/llama3.2",
            api_base="http://localhost:11434",
            temperature=0.1,
            max_tokens=150
        )
        
        # Set the LLM as the default model for DSPy
        dspy.settings.configure(lm=llm)
        
    except Exception as e:
        print(f"Error configuring Ollama: {e}")
        print("Trying alternative configuration...")
        
        # Alternative configuration approach
        try:
            llm = dspy.LM(model="ollama_chat/llama3.2")
            dspy.settings.configure(lm=llm)
        except Exception as e2:
            print(f"Alternative configuration failed: {e2}")
            raise Exception("Could not configure Ollama with DSPy")
    
    class TopicAnalyzer(dspy.Signature):
        """Analyze text and determine if it belongs to the current topic or starts a new one."""
        
        context = dspy.InputField(desc="Previous paragraphs for context")
        current_text = dspy.InputField(desc="Current paragraph to analyze")
        current_chunk = dspy.InputField(desc="Current chunk number")
        
        decision = dspy.OutputField(desc="Answer 'NEW' to start new chunk or 'CONTINUE' to stay in current chunk")
        reason = dspy.OutputField(desc="Brief reason for decision")
    
    class ChunkTitleGenerator(dspy.Signature):
        """Generate a title and description for a new chunk."""
        
        chunk_content = dspy.InputField(desc="Content of the new chunk")
        
        title = dspy.OutputField(desc="Short, descriptive title for the chunk (max 3 words)")
        description = dspy.OutputField(desc="Brief description of what this chunk covers (max 10 words)")
    
    class ChunkAnalyzer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.analyzer = dspy.ChainOfThought(TopicAnalyzer)
            self.title_generator = dspy.ChainOfThought(ChunkTitleGenerator)
        
        def forward(self, context, current_text, current_chunk):
            try:
                result = self.analyzer(
                    context=context,
                    current_text=current_text,
                    current_chunk=current_chunk
                )
                return result.decision, result.reason
            except Exception as e:
                print(f"LLM call failed: {e}")
                # Fallback logic
                return "CONTINUE", "LLM error - using fallback"
        
        def generate_title_description(self, chunk_content):
            try:
                result = self.title_generator(chunk_content=chunk_content)
                return result.title, result.description
            except Exception as e:
                print(f"Title generation failed: {e}")
                return "Untitled Chunk", "No description available"
    
    # Initialize the analyzer
    analyzer = ChunkAnalyzer()
    
    def format_paragraph_for_context(para: Dict[str, Any]) -> str:
        """Format a paragraph's metadata and content for LLM context."""
        para_type = para.get('type', 'text')
        level = para.get('level', 0)
        content = para.get('md-text', '').strip()[:100]  # Limit content length
        
        if para_type == 'header':
            return f"HEADER(L{level}): {content}"
        else:
            return f"{para_type.upper()}: {content}"
    
    def is_paragraph_continuation(current_para: Dict[str, Any], 
                                 previous_para: Dict[str, Any]) -> bool:
        """
        Check if the current paragraph is a continuation of the previous paragraph.
        
        Args:
            current_para (Dict[str, Any]): Current paragraph to check
            previous_para (Dict[str, Any]): Previous paragraph to compare against
            
        Returns:
            bool: True if current paragraph appears to be a continuation
        """
        # Skip headers, tables, and other non-text elements
        # Note: Headers should never be continuations - they always start new chunks
        if (current_para.get('type') in ['header', 'table', 'image', 'code', 'quote', 'list', 'latex'] or
            previous_para.get('type') in ['header', 'table', 'image', 'code', 'quote', 'list', 'latex']):
            return False
        
        current_text = current_para.get('md-text', '').strip()
        previous_text = previous_para.get('md-text', '').strip()
        
        if not current_text or not previous_text:
            return False
        
        # Remove markdown formatting for better text analysis
        import re
        current_clean = re.sub(r'[`*_\[\]()]', '', current_text)
        previous_clean = re.sub(r'[`*_\[\]()]', '', previous_text)
        
        # Check if current paragraph starts with lowercase (continuation indicator)
        if current_clean and current_clean[0].islower():
            return True
        
        # Check if previous paragraph doesn't end with sentence-ending punctuation
        sentence_endings = ['.', '!', '?', ':', ';']
        if previous_clean and not any(previous_clean.endswith(ending) for ending in sentence_endings):
            return True
        
        # Additional checks for common continuation patterns
        # Check for incomplete sentences ending with common continuation words
        continuation_words = ['and', 'or', 'but', 'however', 'therefore', 'thus', 'hence', 'so', 'because', 'since', 'while', 'although', 'though']
        previous_words = previous_clean.lower().split()
        if previous_words and previous_words[-1] in continuation_words:
            return True
        
        # Check if current paragraph starts with common continuation words
        current_words = current_clean.lower().split()
        if current_words and current_words[0] in continuation_words:
            return True
        
        return False
    
    def should_create_new_chunk(new_para: Dict[str, Any], 
                               context_paragraphs: List[Dict[str, Any]], 
                               current_chunk: int) -> tuple[bool, str]:
        """Determine if a new paragraph should start a new chunk."""
        
        # Always create new chunk for headers (they signal new topics/sections)
        if new_para.get('type') == 'header':
            level = new_para.get('level', 0)
            return True, f"Header (L{level}) - new topic/section"
        
        # Check for paragraph continuation first
        if context_paragraphs:
            previous_para = context_paragraphs[-1]
            if is_paragraph_continuation(new_para, previous_para):
                return False, "Paragraph continuation detected"
        
        #   Example:
        #   Previous context:
        #    [HEADER L1] # Heart Transplant
        #    [TEXT] One of the possible complications...
        #    [TEXT] Antibodies are found in your blood...

        #    Current paragraph to analyze:
        #    [TEXT] This is a regular paragraph that might continue the topic.

        #    Current chunk number: 1

        #    Analyze if this paragraph belongs to the current topic or starts a new one.
        #    Answer 'NEW' to start new chunk or 'CONTINUE' to stay in current chunk.
        #    Provide brief reasoning.
        
        # Format context
        if context_paragraphs:
            context = "Previous context:\n" + "\n".join([
                format_paragraph_for_context(p) 
                for p in context_paragraphs[-5:]  # Last 5 paragraphs
            ])
        else:
            context = "No previous context"
        
        # Format current paragraph
        current_text = format_paragraph_for_context(new_para)
        
        # Get LLM decision
        decision, reason = analyzer(
            context=context,
            current_text=current_text,
            current_chunk=str(current_chunk)
        )
        
        # Parse decision
        create_new = decision.upper().strip() in ['NEW', 'TRUE', 'YES', '1']
        
        return create_new, reason
    
    # Process all windows and assign chunk numbers
    enhanced_paragraphs = []
    current_chunk = 1
    context_paragraphs = []
    processed_texts = set()  # Track processed paragraphs to avoid duplicates
    current_chunk_title = "Introduction"
    current_chunk_description = "Initial content of the document"
    
    print("Starting semantic chunk analysis with Ollama...")
    print(f"Total windows to process: {len(windows)}")
    
    # Process all paragraphs from all windows
    for i, window in enumerate(windows):
        if i % 20 == 0:
            print(f"Progress: {i+1}/{len(windows)} windows")
        
        # Process all paragraphs in the current window
        for para in window:
            para_text = para.get('md-text', '')
            
            # Skip if already processed
            if para_text in processed_texts:
                continue
            
            # For the first paragraph, start with chunk 1
            if len(enhanced_paragraphs) == 0:
                should_create_new = False
                reasoning = "First paragraph"
            else:
                # Analyze if this paragraph should start a new chunk
                should_create_new, reasoning = should_create_new_chunk(
                    para, 
                    context_paragraphs, 
                    current_chunk
                )
                
                # Show continuation detection info
                if reasoning == "Paragraph continuation detected":
                    print(f"  🔗 CONTINUATION: {para_text[:60]}...")
                elif reasoning.startswith("Header (L"):
                    print(f"  📋 HEADER: {para_text[:60]}...")
            
            # Create enhanced paragraph with chunk number
            para_enhanced = para.copy()
            if should_create_new:
                current_chunk += 1
                print(f"  NEW CHUNK {current_chunk}: {para_text[:60]}...")
                print(f"  Reason: {reasoning}")
                
                # Generate title and description for new chunk
                chunk_content = para_text[:200]  # Use first 200 chars for title generation
                current_chunk_title, current_chunk_description = analyzer.generate_title_description(chunk_content)
            
            para_enhanced['chunk_number'] = current_chunk
            para_enhanced['chunk_title'] = current_chunk_title
            para_enhanced['chunk_description'] = current_chunk_description
            enhanced_paragraphs.append(para_enhanced)
            context_paragraphs.append(para_enhanced)
            processed_texts.add(para_text)
            
            # Keep context manageable
            if len(context_paragraphs) > 5:
                context_paragraphs.pop(0)
            
            # Write to JSON every 20 chunks if output filename is provided
            if output_filename and current_chunk % 20 == 0:
                try:
                    # Create temporary JSON with current progress
                    temp_json = convert_chunks_to_json(
                        paragraphs=enhanced_paragraphs,
                        output_file=output_filename,
                        format_type="chunked"
                    )
                    print(f"  📝 Saved progress to {output_filename} (chunk {current_chunk})")
                except Exception as e:
                    print(f"  ⚠️  Warning: Failed to save progress to {output_filename}: {e}")
    
    print(f"\nCompleted analysis!")
    print(f"Created {current_chunk} semantic chunks from {len(enhanced_paragraphs)} paragraphs.")
    
    # Final save if output filename is provided
    if output_filename:
        try:
            final_json = convert_chunks_to_json(
                paragraphs=enhanced_paragraphs,
                output_file=output_filename,
                format_type="chunked"
            )
            print(f"📄 Final results saved to {output_filename}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save final results to {output_filename}: {e}")
    
    return enhanced_paragraphs

def convert_chunks_to_json(paragraphs: List[Dict[str, Any]], 
                          output_file: str = None,
                          include_metadata: bool = True,
                          format_type: str = "structured") -> str:
    """
    Convert chunked paragraphs into a JSON text document.
    
    Args:
        paragraphs (List[Dict[str, Any]]): List of enhanced paragraphs with chunk information
        output_file (str): Optional file path to save the JSON (if None, returns JSON string)
        include_metadata (bool): Whether to include all metadata fields
        format_type (str): JSON structure type - "structured", "simple", or "chunked"
        
    Returns:
        str: JSON string representation of the chunked paragraphs
    """
    
    def create_structured_json():
        """Create structured JSON with detailed metadata."""
        document = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_paragraphs": len(paragraphs),
                "total_chunks": len(set(p.get('chunk_number', 0) for p in paragraphs)),
                "format_version": "1.0"
            },
            "chunks": {}
        }
        
        # Group paragraphs by chunk
        for para in paragraphs:
            chunk_num = para.get('chunk_number', 0)
            if chunk_num not in document["chunks"]:
                document["chunks"][chunk_num] = {
                    "chunk_number": chunk_num,
                    "title": para.get('chunk_title', 'Untitled'),
                    "description": para.get('chunk_description', 'No description'),
                    "paragraphs": []
                }
            
            # Create paragraph object
            para_obj = {
                "content": para.get('md-text', ''),
                "type": para.get('type', 'text'),
                "level": para.get('level', 0)
            }
            
            if include_metadata:
                para_obj.update({
                    "line_number": para.get('line_number', 0),
                    "chunk_title": para.get('chunk_title', ''),
                    "chunk_description": para.get('chunk_description', '')
                })
            
            document["chunks"][chunk_num]["paragraphs"].append(para_obj)
        
        return document
    
    def create_simple_json():
        """Create simple JSON with minimal structure."""
        document = {
            "created_at": datetime.now().isoformat(),
            "paragraphs": []
        }
        
        for para in paragraphs:
            para_obj = {
                "content": para.get('md-text', ''),
                "chunk_number": para.get('chunk_number', 0),
                "chunk_title": para.get('chunk_title', ''),
                "chunk_description": para.get('chunk_description', '')
            }
            
            if include_metadata:
                para_obj.update({
                    "type": para.get('type', 'text'),
                    "level": para.get('level', 0)
                })
            
            document["paragraphs"].append(para_obj)
        
        return document
    
    def create_chunked_json():
        """Create JSON organized by chunks with summary statistics."""
        # Get chunk statistics
        chunks = {}
        for para in paragraphs:
            chunk_num = para.get('chunk_number', 0)
            if chunk_num not in chunks:
                chunks[chunk_num] = {
                    "chunk_number": chunk_num,
                    "title": para.get('chunk_title', 'Untitled'),
                    "description": para.get('chunk_description', 'No description'),
                    "paragraphs": [],
                    "statistics": {
                        "total_paragraphs": 0,
                        "header_count": 0,
                        "text_count": 0,
                        "other_count": 0
                    }
                }
            
            chunks[chunk_num]["paragraphs"].append({
                "content": para.get('md-text', ''),
                "type": para.get('type', 'text'),
                "level": para.get('level', 0)
            })
            
            # Update statistics
            chunks[chunk_num]["statistics"]["total_paragraphs"] += 1
            para_type = para.get('type', 'text')
            if para_type == 'header':
                chunks[chunk_num]["statistics"]["header_count"] += 1
            elif para_type == 'text':
                chunks[chunk_num]["statistics"]["text_count"] += 1
            else:
                chunks[chunk_num]["statistics"]["other_count"] += 1
        
        document = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_paragraphs": len(paragraphs),
                "total_chunks": len(chunks),
                "format_version": "1.0"
            },
            "chunks": list(chunks.values())
        }
        
        return document
    
    # Create JSON based on format type
    if format_type == "structured":
        document = create_structured_json()
    elif format_type == "simple":
        document = create_simple_json()
    elif format_type == "chunked":
        document = create_chunked_json()
    else:
        raise ValueError("format_type must be 'structured', 'simple', or 'chunked'")
    
    # Convert to JSON string
    json_string = json.dumps(document, indent=2, ensure_ascii=False)
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_string)
            print(f"JSON saved to: {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
    
    return json_string

# Example usage functions:
def save_chunks_as_json(paragraphs: List[Dict[str, Any]], 
                       filename: str = None,
                       format_type: str = "chunked") -> str:
    """
    Convenience function to save chunked paragraphs as JSON.
    
    Args:
        paragraphs (List[Dict[str, Any]]): Enhanced paragraphs with chunk information
        filename (str): Output filename (if None, auto-generates)
        format_type (str): JSON format type
        
    Returns:
        str: Path to saved JSON file
    """
    
    if filename is None:
        # Auto-generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chunked_paragraphs_{timestamp}.json"
    
    json_string = convert_chunks_to_json(
        paragraphs=paragraphs,
        output_file=filename,
        format_type=format_type
    )
        
    return filename

def print_latex_aware_chunks(windows: List[List[Dict[str, Any]]], 
                           show_latex_details: bool = True,
                           max_chars: int = 80):
    """
    Print LaTeX-aware chunks with detailed LaTeX information.
    
    Args:
        windows (List[List[Dict[str, Any]]]): List of window groups
        show_latex_details (bool): Whether to show detailed LaTeX information
        max_chars (int): Maximum characters to show per paragraph
    """
    
    print(f"Created {len(windows)} LaTeX-aware chunks:\n")
    
    for i, window in enumerate(windows, 1):
        print(f"Chunk {i}:")
        
        for j, para in enumerate(window, 1):
            content = para.get('md-text', '')
            para_type = para.get('type', 'unknown')
            
            if para_type == 'latex':
                latex_info = para.get('latex_info', {})
                latex_type = latex_info.get('latex_type', 'unknown')
                
                print(f"  P{j}: ({para_type}:{latex_type}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
                
                if show_latex_details and latex_info.get('has_latex'):
                    # Show LaTeX details
                    if latex_info.get('inline_math'):
                        print(f"    Inline math: {len(latex_info['inline_math'])} expressions")
                    if latex_info.get('display_math'):
                        print(f"    Display math: {len(latex_info['display_math'])} expressions")
                    if latex_info.get('environments'):
                        print(f"    Environments: {', '.join(latex_info['environments'])}")
                    if latex_info.get('commands'):
                        print(f"    Commands: {', '.join(latex_info['commands'])}")
            else:
                level = para.get('level', 0)
                if para_type == 'header' and level > 0:
                    print(f"  P{j}: ({para_type} L{level}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
                else: 
                    print(f"  P{j}: ({para_type}) {content[:max_chars]}{'...' if len(content) > max_chars else ''}")
        
        print()


if __name__ == "__main__":
    # Example usage
    paragraphs_advanced = read_markdown_paragraphs_advanced(sys.argv[1])
    raw_windows = process_paragraphs_fifo_for_llm(paragraphs_advanced, window_size=5, step_size=1)
    
    # Generate output filename
    output_filename = "semantically_chunked_paragraphs_" + filename.split(".md")[0] + ".json"
    
    # Process with incremental JSON writing every 20 chunks
    semantically_chunked_paragraphs = llm_for_context_chunking(raw_windows, output_filename=output_filename)