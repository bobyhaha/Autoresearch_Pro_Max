"""Queue-driven, provenance-first autoresearch pipeline."""

from .campaign import CampaignQueue
from .evidence import EvidenceEngine
from .execution import ExecutionService
from .knowledge import KnowledgeEngine
from .research import ResearchEngine
from .science import ScientificLibrary
from .sealing import SealingAuthority
from .store import Store
from .workflow import V2Workflow

__all__ = [
    "CampaignQueue",
    "EvidenceEngine",
    "ExecutionService",
    "KnowledgeEngine",
    "ResearchEngine",
    "ScientificLibrary",
    "SealingAuthority",
    "Store",
    "V2Workflow",
]

__version__ = "0.3.0"
