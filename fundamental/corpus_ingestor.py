"""
Corpus Ingestor
===============

Ingests PH Macro transcripts into vector database for semantic search.

ARCHITECTURE:
- Sentence-transformers for embeddings (all-MiniLM-L6-v2)
- ChromaDB for vector storage
- Document chunking with overlap
- Metadata tagging (date, speaker, topic cluster)

USAGE:
    python -m fundamental.corpus_ingestor --rebuild
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("⚠️  Missing dependencies. Install:")
    print("    pip install sentence-transformers chromadb")
    exit(1)


class CorpusIngestor:
    """Ingests PH Macro transcripts into ChromaDB vector store"""
    
    def __init__(self, db_path: str = "./data/chroma_db"):
        """
        Initialize corpus ingestor.
        
        Args:
            db_path: Path to ChromaDB persistent storage
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        print("Loading sentence-transformers model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="ph_macro_corpus",
            metadata={"description": "PH Macro transcript embeddings"}
        )
        
        print(f"✓ Corpus ready. Documents in collection: {self.collection.count()}")
    
    def ingest_transcripts(self, transcripts_dir: str = "./transcripts/processed"):
        """
        Ingest all processed transcripts from directory.
        
        Args:
            transcripts_dir: Path to processed transcript JSON files
        """
        transcripts_path = Path(transcripts_dir)
        
        if not transcripts_path.exists():
            print(f"❌ Transcripts directory not found: {transcripts_path}")
            return
        
        json_files = list(transcripts_path.glob("*.json"))
        
        if len(json_files) == 0:
            print(f"⚠️  No JSON files found in {transcripts_path}")
            return
        
        print(f"\n{'='*70}")
        print(f"INGESTING {len(json_files)} TRANSCRIPT FILES")
        print(f"{'='*70}\n")
        
        total_chunks = 0
        
        for json_file in json_files:
            print(f"Processing: {json_file.name}...")
            
            with open(json_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Extract metadata
            video_id = transcript_data.get('video_id', 'unknown')
            title = transcript_data.get('title', 'Unknown Title')
            upload_date = transcript_data.get('upload_date', 'unknown')
            
            # Get text content
            text = transcript_data.get('text', '')
            
            if not text:
                print(f"  ⚠️  No text content, skipping")
                continue
            
            # Chunk text (500 chars with 100 char overlap)
            chunks = self._chunk_text(text, chunk_size=500, overlap=100)
            
            # Generate embeddings
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            
            # Prepare for ChromaDB
            ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    'video_id': video_id,
                    'title': title,
                    'upload_date': upload_date,
                    'chunk_idx': i,
                    'total_chunks': len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            
            total_chunks += len(chunks)
            print(f"  → Added {len(chunks)} chunks")
        
        print(f"\n{'='*70}")
        print(f"✓ INGESTION COMPLETE")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Collection size: {self.collection.count()}")
        print(f"{'='*70}\n")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c) > 50]  # Filter very short chunks
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search in corpus.
        
        Args:
            query: Search query string
            n_results: Number of results to return
        
        Returns:
            List of result dicts with {text, metadata, distance}
        """
        query_embedding = self.model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        search_results = []
        
        for i in range(len(results['ids'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return search_results
    
    def rebuild(self, transcripts_dir: str = "./transcripts/processed"):
        """Rebuild collection from scratch"""
        print("\n🔄 Rebuilding corpus...")
        
        # Delete existing collection
        try:
            self.client.delete_collection("ph_macro_corpus")
            print("  Deleted old collection")
        except:
            pass
        
        # Recreate
        self.collection = self.client.get_or_create_collection(
            name="ph_macro_corpus",
            metadata={"description": "PH Macro transcript embeddings"}
        )
        
        # Ingest
        self.ingest_transcripts(transcripts_dir)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest PH Macro transcripts into vector DB")
    parser.add_argument('--rebuild', action='store_true', help='Rebuild collection from scratch')
    parser.add_argument('--search', type=str, help='Test semantic search')
    parser.add_argument('--transcripts-dir', type=str, default='./transcripts/processed',
                       help='Path to processed transcript JSON files')
    
    args = parser.parse_args()
    
    ingestor = CorpusIngestor()
    
    if args.rebuild:
        ingestor.rebuild(args.transcripts_dir)
    
    elif args.search:
        print(f"\n🔍 Searching for: '{args.search}'\n")
        results = ingestor.search(args.search, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Distance: {result['distance']:.3f}")
            print(f"    Video: {result['metadata']['title']}")
            print(f"    Date: {result['metadata']['upload_date']}")
            print(f"    Text: {result['text'][:200]}...")
            print()
    
    else:
        print("\nUsage:")
        print("  python -m fundamental.corpus_ingestor --rebuild")
        print("  python -m fundamental.corpus_ingestor --search 'inflation outlook'")
