from typing import Any, Dict, TypedDict, List


class IngestState(TypedDict, total=False):
    """
    Represents the state of a contract, in processing order
    """
    # initial input data
    file_path: str
    file_type: str      # pdf/txt/image
    
    # preprocessing
    temp_dir: str
    page_image_paths: List[str]
    
    # OCR / Text extract
    raw_text_by_page: Dict[int, str]  # page number to text mapping
    full_text: str
    
    # Indexing
    collection_name: str  # in: target collection; defaults to DEFAULT_COLLECTION_NAME
    document_id: str      # in/out: identity within the collection, generated if absent
    index_key: str        # in/out: identity of the shared vector artifact
    source_name: str      # in: display name for citations; defaults to the filename
    chunks: List[str]
    chunk_metadata: List[Dict[str, Any]]  # metadata for each chunk
    index_ready: bool
    document_hash: str    # out: the hash the chunks were indexed under
    
    # extracted fields
    extracted_fields: Dict[str, str]
    
    review_comments: List[str]
    is_approved: bool
    
    # output 
    excel_output_path: str
    
    # metadata
    current_step: str
    processing_log: List[str]