"""
Module 2: Relation-Level DAG Provenance Auditing (Primary Detector)
Corresponds to Section 2.2, Equation (3), and Algorithm 1 in STRAND (NAACL 2026).

Mathematical Formulation:
    CausalSupp(u -> v, S) = I[ exists s_i in S_prem : Cooccur(u, v, s_i) and HasConn(s_i) ]
    Topological Provenance Ratio:
        eta(G) = (1 / |E|) * sum_{e in E} Gr(e)
    Pruning Condition:
        eta(G) < eta_thresh (calibrated: 0.50)
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

try:
    import networkx as nx
except ImportError:
    class _SimpleDiGraph:
        def __init__(self):
            self.nodes = {}
            self._edges = []
        def add_node(self, node_id, **attrs):
            self.nodes[node_id] = attrs
        def add_edge(self, u, v, **attrs):
            self._edges.append((u, v, attrs))
        def edges(self, data=False):
            return self._edges
    class _NXMock:
        DiGraph = _SimpleDiGraph
    nx = _NXMock()


class TypedRelation(str, Enum):
    """Relation categories defined in Section 2.2 and Algorithm 1."""
    CAUSAL_SUPPORT = "CAUSAL-SUPPORT"
    TEMPORAL_ONLY = "TEMPORAL-ONLY"
    CORRELATIONAL = "CORRELATIONAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class CausalSuppVerdict:
    """Individual edge audit decision returned by CausalSupp."""
    source_clause: str
    target_clause: str
    is_grounded: int          # Gr(e) in {0, 1}
    relation_type: TypedRelation
    matching_premise: Optional[str] = None


@dataclass
class DAGAuditReport:
    """Graph-level audit results across candidate reasoning DAG G = (V, E)."""
    total_edges: int
    grounded_edges: int
    unsupported_edges: int
    provenance_ratio: float   # eta(G)
    is_manipulated: bool      # True if eta(G) < eta_thresh
    edge_verdicts: List[CausalSuppVerdict] = field(default_factory=list)
    pruned_edges: List[Tuple[str, str]] = field(default_factory=list)


# Calibrated connector lexicons (Algorithm 1)
DEFAULT_CAUSAL_CONNECTORS: Set[str] = {
    "because",
    "therefore",
    "as a result",
    "caused",
    "led to",
    "leads to",
    "proves that",
    "proving that",
    "indicates that",
    "indicating that",
    "consequently",
    "thus",
    "hence",
}

DEFAULT_TEMPORAL_CONNECTORS: Set[str] = {
    "later",
    "subsequently",
    "after",
    "before",
    "then",
    "followed by",
    "meanwhile",
    "concurrently",
    "prior to",
}

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its"
}


def extract_keywords(clause: str) -> Set[str]:
    """Extract content keywords W from clause (Line 1 of Algorithm 1)."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clause.lower())
    return {t for t in tokens if t not in STOPWORDS}


def has_causal_connector(premise_text: str, lexicon: Set[str]) -> bool:
    """Check if premise contains an explicit causal connector (Line 4 of Algorithm 1)."""
    text_lower = premise_text.lower()
    for conn in lexicon:
        if re.search(rf"\b{re.escape(conn)}\b", text_lower):
            return True
    return False


def has_temporal_connector(premise_text: str, lexicon: Set[str]) -> bool:
    """Check if premise contains temporal ordering connectors (Line 6 of Algorithm 1)."""
    text_lower = premise_text.lower()
    for conn in lexicon:
        if re.search(rf"\b{re.escape(conn)}\b", text_lower):
            return True
    return False


def causal_supp(
    edge: Tuple[str, str],
    premises: List[str],
    causal_lexicon: Optional[Set[str]] = None,
    temporal_lexicon: Optional[Set[str]] = None
) -> CausalSuppVerdict:
    """
    Implements Algorithm 1: Deterministic CausalSupp Edge Provenance Audit.
    
    Checks whether candidate edge e = (u -> v) has explicit source grounding
    in premise set S_prem.
    """
    u, v = edge
    c_lex = causal_lexicon or DEFAULT_CAUSAL_CONNECTORS
    t_lex = temporal_lexicon or DEFAULT_TEMPORAL_CONNECTORS

    w_u = extract_keywords(u)
    w_v = extract_keywords(v)

    # Line 2: for each premise s_i in S_prem do
    for s_i in premises:
        s_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", s_i.lower()))
        
        # Line 3: if W_u subseteq s_i and W_v subseteq s_i then
        u_match = bool(w_u and w_u.issubset(s_tokens))
        v_match = bool(w_v and w_v.issubset(s_tokens))

        # Check soft overlap if strict subset is empty due to phrasing
        if not (u_match and v_match):
            u_overlap = len(w_u & s_tokens) / max(len(w_u), 1)
            v_overlap = len(w_v & s_tokens) / max(len(w_v), 1)
            if u_overlap >= 0.50 and v_overlap >= 0.50:
                u_match, v_match = True, True

        if u_match and v_match:
            # Line 4: if exists c in C_causal s.t. c in s_i then
            if has_causal_connector(s_i, c_lex):
                return CausalSuppVerdict(
                    source_clause=u,
                    target_clause=v,
                    is_grounded=1,
                    relation_type=TypedRelation.CAUSAL_SUPPORT,
                    matching_premise=s_i
                )
            # Line 6: else if HasTemporalConnector(s_i) then
            elif has_temporal_connector(s_i, t_lex):
                return CausalSuppVerdict(
                    source_clause=u,
                    target_clause=v,
                    is_grounded=0,
                    relation_type=TypedRelation.TEMPORAL_ONLY,
                    matching_premise=s_i
                )
            # Line 8: else
            else:
                return CausalSuppVerdict(
                    source_clause=u,
                    target_clause=v,
                    is_grounded=0,
                    relation_type=TypedRelation.CORRELATIONAL,
                    matching_premise=s_i
                )

    # Line 13: return (Gr(e) = 0, R(e) = UNSUPPORTED)
    return CausalSuppVerdict(
        source_clause=u,
        target_clause=v,
        is_grounded=0,
        relation_type=TypedRelation.UNSUPPORTED,
        matching_premise=None
    )


class CausalDAGAuditor:
    """
    Parses candidate reasoning R into a directed acyclic graph G = (V, E)
    and computes the topological provenance ratio eta(G) using CausalSupp.
    """

    def __init__(
        self,
        eta_thresh: float = 0.50,
        causal_lexicon: Optional[Set[str]] = None,
        temporal_lexicon: Optional[Set[str]] = None
    ) -> None:
        """
        Initialize the DAG Auditor.
        
        Args:
            eta_thresh: Provenance threshold ratio (calibrated: 0.50).
            causal_lexicon: Explicit causal connectors C_causal.
            temporal_lexicon: Temporal sequence connectors.
        """
        self.eta_thresh = float(eta_thresh)
        self.causal_lexicon = causal_lexicon or DEFAULT_CAUSAL_CONNECTORS
        self.temporal_lexicon = temporal_lexicon or DEFAULT_TEMPORAL_CONNECTORS

    def parse_reasoning_to_dag(self, rationale_text: str) -> nx.DiGraph:
        """
        Parse natural language Chain-of-Thought reasoning into directed graph G = (V, E).
        Decomposes text into atomic propositional clauses and sequential inferential edges.
        """
        g = nx.DiGraph()
        clauses = [
            c.strip()
            for c in re.split(r"[.;\n]+", rationale_text)
            if len(c.strip()) > 10
        ]

        if not clauses:
            return g

        for i, clause in enumerate(clauses):
            g.add_node(i, text=clause)
            if i > 0:
                # Directed inferential edge representing reasoning transition u -> v
                g.add_edge(i - 1, i, source=clauses[i - 1], target=clause)

        return g

    def audit(
        self,
        rationale_text: str,
        premises: List[str]
    ) -> DAGAuditReport:
        """
        Audit all inferential edges in G against verified source premises.
        
        Returns:
            DAGAuditReport containing eta(G), grounded count, and pruned edges.
        """
        g = self.parse_reasoning_to_dag(rationale_text)
        edges = list(g.edges(data=True))

        if not edges:
            return DAGAuditReport(
                total_edges=0,
                grounded_edges=0,
                unsupported_edges=0,
                provenance_ratio=1.0,
                is_manipulated=False
            )

        verdicts: List[CausalSuppVerdict] = []
        grounded_count = 0
        pruned_edges: List[Tuple[str, str]] = []

        for u_idx, v_idx, data in edges:
            u_text = data.get("source", g.nodes[u_idx].get("text", ""))
            v_text = data.get("target", g.nodes[v_idx].get("text", ""))

            verdict = causal_supp(
                edge=(u_text, v_text),
                premises=premises,
                causal_lexicon=self.causal_lexicon,
                temporal_lexicon=self.temporal_lexicon
            )
            verdicts.append(verdict)

            if verdict.is_grounded == 1:
                grounded_count += 1
            else:
                pruned_edges.append((u_text, v_text))

        total = len(edges)
        eta = grounded_count / float(total) if total > 0 else 1.0
        is_manipulated = bool(eta < self.eta_thresh)

        return DAGAuditReport(
            total_edges=total,
            grounded_edges=grounded_count,
            unsupported_edges=total - grounded_count,
            provenance_ratio=float(eta),
            is_manipulated=is_manipulated,
            edge_verdicts=verdicts,
            pruned_edges=pruned_edges
        )
