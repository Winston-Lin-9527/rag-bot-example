from typing import Any, Dict, TypedDict, List


class ContractState(TypedDict, total=False):
    """
    Represents the state of a contract, in processing order
    """
    # initial input data
    file_path: str
    file_type: str      # pdf/txt/image
    
    # preprocessing
    page_image_paths: List[str]
    
    # OCR / Text extract
    raw_text_by_page: Dict[int, str]  # page number to text mapping
    full_text: str
    
    # Indexing
    chunks: List[str]
    chunk_metadata: List[Dict[str, Any]]  # metadata for each chunk
    index_ready: bool
    
    # extracted fields
    extracted_fields: Dict[str, str]
    
    review_comments: List[str]
    is_approved: bool
    
    # output 
    excel_output_path: str
    
    # metadata
    current_step: str
    processing_log: List[str]