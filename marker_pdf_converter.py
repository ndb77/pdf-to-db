#!/usr/bin/env python3
"""
PDF to Markdown Converter using Marker
======================================

This script uses the marker library to convert PDF documents to high-quality
markdown format with text lossless conversion. Marker is specifically designed
for accurate PDF to markdown conversion using deep learning models.

Requirements:
- marker-pdf library
- torch (for model inference)

Usage:
    python marker_pdf_converter.py input.pdf [--output output.md] [--use-llm] [--force-ocr]
"""

import os
import argparse
import json
import sys
import time
import traceback
from typing import Dict, Any
import textwrap

# Add better error handling for imports
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError as exc:
    print(f"ERROR: Missing required Marker library: {exc}")
    print("Please install with: pip install marker-pdf")
    print("Or activate the virtual environment: marker-venv\\Scripts\\Activate.ps1")
    MARKER_AVAILABLE = False

try:
    import dspy
    import litellm
    DSPY_AVAILABLE = True
except ImportError as exc:
    print(f"Warning: {exc}. Ollama integration will not be available.")
    dspy = None
    litellm = None
    DSPY_AVAILABLE = False

# Check if we're in the right environment
if not MARKER_AVAILABLE:
    sys.exit(1)


class MarkerJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable objects from Marker."""
    
    def default(self, obj):
        # Handle PIL Images
        if hasattr(obj, 'mode') and hasattr(obj, 'size'):
            return f"<Image: {obj.size[0]}x{obj.size[1]} {getattr(obj, 'mode', 'unknown')}>"
        
        # Handle other non-serializable objects
        if hasattr(obj, '__dict__'):
            try:
                return obj.__dict__
            except:
                return f"<{type(obj).__name__}: non-serializable>"
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
            
        # Handle other types
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__}: non-serializable>"


def clean_data_for_json(data):
    """Recursively clean data to make it JSON serializable."""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            try:
                # Try to serialize the value to see if it's safe
                json.dumps(value, cls=MarkerJSONEncoder)
                cleaned[key] = clean_data_for_json(value)
            except (TypeError, ValueError):
                # If it fails, convert to string representation
                cleaned[key] = f"<{type(value).__name__}: non-serializable>"
        return cleaned
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, set):
        return list(data)
    else:
        return data


def convert_pdf_with_marker(
    pdf_path: str,
    use_ollama: bool = False,
    ollama_model: str = "llama3.2",
    ollama_endpoint: str = "http://localhost:11434"
) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Convert PDF using Marker with optional Ollama integration for text reconstruction.
    
    Returns:
        tuple: (markdown_text, plain_text, json_data, metadata)
    """
    
    print(f"  • Converting PDF with Marker...")
    
    # Create basic Marker converter without complex configuration for now
    try:
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        # Convert the PDF
        rendered = converter(pdf_path)
        
        # Extract text and images
        text, _, images = text_from_rendered(rendered)
        
        # Get markdown output - Marker's default output is markdown
        markdown_text = text
        
        # Get JSON data and metadata with proper cleaning
        try:
            if hasattr(rendered, 'model_dump'):
                raw_json_data = rendered.model_dump()
            elif hasattr(rendered, 'dict'):
                raw_json_data = rendered.dict()
            else:
                raw_json_data = {
                    "text": text,
                    "type": "conversion_output",
                    "source": "marker"
                }
            
            # Clean the data for JSON serialization
            json_data = clean_data_for_json(raw_json_data)
            
        except Exception as e:
            print(f"  • Warning: Could not extract full JSON data: {e}")
            json_data = {
                "text": text,
                "type": "conversion_output", 
                "source": "marker",
                "error": str(e)
            }
        
        # Try to get metadata
        try:
            metadata = getattr(rendered, 'metadata', {})
            if not isinstance(metadata, dict):
                metadata = {"source": "marker", "status": "converted"}
            # Clean metadata as well
            metadata = clean_data_for_json(metadata)
        except Exception:
            metadata = {"source": "marker", "status": "converted"}
        
        return markdown_text, text, json_data, metadata
        
    except Exception as e:
        print(f"  • Error in Marker conversion: {e}")
        raise


def _build_lm(model: str, api_base: str):
    """Return a DSPy language-model instance pointed at the local Ollama REST API."""
    if not dspy:
        raise ImportError("DSPy not available for Ollama integration")
    
    return dspy.LM(
        model=f"ollama/{model}",
        temperature=0.2,
        max_tokens=4096,
        api_base=api_base,
    )


def apply_ollama_text_reconstruction(
    raw_text: str, 
    model: str, 
    endpoint: str
) -> str:
    """Use Ollama to reconstruct logical reading order and fix sentence breaks."""
    
    if not dspy:
        print("  • Warning: DSPy not available, skipping Ollama reconstruction")
        return raw_text
    
    if not raw_text.strip():
        return raw_text
    
    print("  • Applying Ollama text reconstruction...")
    
    api_base = endpoint.split("/api", 1)[0] if "/api" in endpoint else endpoint.rstrip("/")
    
    try:
        lm = _build_lm(model=model, api_base=api_base)
        
        system_prompt = (
            "You are an expert text reconstruction specialist. The user will provide raw text that "
            "may have reading order issues from a multi-column PDF or document. "
            "Your task is to: "
            "1) Reconstruct the logical reading order "
            "2) Fix obvious sentence breaks and formatting issues "
            "3) Preserve all original content without adding or removing information "
            "4) Maintain proper paragraph structure "
            "5) Keep tables, lists, and special formatting intact "
            "Return ONLY the reconstructed text without any additional commentary."
        )
        
        user_prompt = textwrap.dedent(
            f"""\
            Please reconstruct the following text by fixing reading order and sentence breaks:

            {raw_text}
            """
        )
        
        response = lm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        reconstructed_text = response[0] if isinstance(response, list) else str(response)
        return reconstructed_text.strip()
        
    except Exception as e:
        print(f"  • Warning: Ollama text reconstruction failed: {e}")
        print(f"  • Returning original text")
        return raw_text


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to markdown, text, and JSON using Marker with optional Ollama text reconstruction."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--output-dir", 
        default=None, 
        help="Output directory (defaults to same directory as PDF)"
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama for text reconstruction and improved accuracy",
    )
    parser.add_argument(
        "--ollama-endpoint", 
        default="http://localhost:11434", 
        help="Ollama endpoint URL [default: %(default)s]"
    )
    parser.add_argument(
        "--model", 
        default="llama3.2", 
        help="Ollama model name [default: %(default)s]"
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force conversion even if output files already exist",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        choices=["markdown", "text", "json"],
        default=["markdown", "text", "json"],
        help="Output formats to generate [default: %(default)s]"
    )
    
    args = parser.parse_args()
    
    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Determine output directory and file paths
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    else:
        output_dir = os.path.dirname(pdf_path) or "."
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Define output file paths
    output_files = {}
    if "markdown" in args.output_formats:
        output_files["markdown"] = os.path.join(output_dir, f"{base_name}_marker.md")
    if "text" in args.output_formats:
        output_files["text"] = os.path.join(output_dir, f"{base_name}_marker_text.txt")
    if "json" in args.output_formats:
        output_files["json"] = os.path.join(output_dir, f"{base_name}_marker.json")
        
    # Check if files already exist
    if not args.force_conversion:
        existing_files = [path for path in output_files.values() if os.path.exists(path)]
        if existing_files:
            print(f"Output files already exist. Use --force-conversion to overwrite:")
            for path in existing_files:
                print(f"  • {path}")
            return
    
    print(f"Converting PDF: {pdf_path}")
    
    # Convert PDF using Marker
    try:
        # Use basic Marker conversion first
        markdown_text, plain_text, json_data, metadata = convert_pdf_with_marker(pdf_path)
        
        # Apply Ollama reconstruction if requested
        if args.use_ollama:
            if not dspy:
                print("  • Warning: DSPy not available, skipping Ollama integration")
                reconstructed_text = plain_text
                reconstructed_markdown = markdown_text
            else:
                reconstructed_text = apply_ollama_text_reconstruction(
                    plain_text, 
                    args.model, 
                    args.ollama_endpoint
                )
                # Also reconstruct markdown if it's different from plain text
                if markdown_text != plain_text:
                    reconstructed_markdown = apply_ollama_text_reconstruction(
                        markdown_text,
                        args.model,
                        args.ollama_endpoint
                    )
                else:
                    reconstructed_markdown = reconstructed_text
        else:
            reconstructed_text = plain_text
            reconstructed_markdown = markdown_text
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
    
    # Save outputs based on requested formats
    saved_files = []
    
    # Save markdown output
    if "markdown" in args.output_formats:
        try:
            with open(output_files["markdown"], "w", encoding="utf-8") as f:
                f.write(reconstructed_markdown)
            print(f"  • Markdown output saved to: {output_files['markdown']}")
            saved_files.append(output_files["markdown"])
        except Exception as e:
            print(f"Error saving markdown: {e}")
    
    # Save text output
    if "text" in args.output_formats:
        try:
            with open(output_files["text"], "w", encoding="utf-8") as f:
                f.write(reconstructed_text)
            print(f"  • Text output saved to: {output_files['text']}")
            saved_files.append(output_files["text"])
        except Exception as e:
            print(f"Error saving text: {e}")
    
    # Save JSON output
    if "json" in args.output_formats:
        try:
            with open(output_files["json"], "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, cls=MarkerJSONEncoder)
            print(f"  • JSON output saved to: {output_files['json']}")
            saved_files.append(output_files["json"])
        except Exception as e:
            print(f"Error saving JSON: {e}")
    
    # Print summary
    print(f"\nConversion complete!")
    print(f"Files generated:")
    for file_path in saved_files:
        print(f"  • {file_path}")
    
    if metadata and isinstance(metadata, dict):
        print(f"\nDocument metadata:")
        for key, value in metadata.items():
            if key == "page_stats" and isinstance(value, list):
                print(f"  • Pages processed: {len(value)}")
            elif key not in ["page_stats"]:  # Skip complex nested data
                print(f"  • {key}: {value}")


if __name__ == "__main__":
    main() 