"""GroundProof — RAG that knows when facts expire, and pays only for the context it needs.

Stage 3 of the Proof series. Built on the AgentProof runtime, graded by its eval harness.

The names below are the headline API — the concepts the article narrative is
built around. Subpackages (``groundproof.ingest``, ``.retrieval``, ``.grading``,
``.compress``, ``.steps``, ``.evals``) carry the full surface; the Chroma
backend stays behind an explicit import (``groundproof.retrieval.chroma_store``)
so importing this package never pays chromadb's startup cost.
"""

from groundproof.compress import CompressedEvidence, QueryAwarePruner
from groundproof.errors import FetchFailure, GroundProofError
from groundproof.grading import EvidenceGrade, EvidenceGrader
from groundproof.ingest import Chunk, Document
from groundproof.retrieval import (
    RankedChunk,
    ResolvedEvidence,
    TemporalRetriever,
    resolve_supersedence,
)
from groundproof.steps import GroundState, build_pipeline

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "CompressedEvidence",
    "Document",
    "EvidenceGrade",
    "EvidenceGrader",
    "FetchFailure",
    "GroundProofError",
    "GroundState",
    "QueryAwarePruner",
    "RankedChunk",
    "ResolvedEvidence",
    "TemporalRetriever",
    "__version__",
    "build_pipeline",
    "resolve_supersedence",
]
