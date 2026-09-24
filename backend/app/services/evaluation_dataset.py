"""
Evaluation dataset for retrieval benchmarking.

Each query contains manually verified relevant chunk IDs
from the indexed research document.
"""

EVALUATION_DATASET = [
    {
        "query": "agentic software engineering",
        "relevant_chunk_ids": [1, 5, 10, 13],
    },
    {
        "query": "definition of agentic software engineering",
        "relevant_chunk_ids": [5, 10, 11],
    },
    {
        "query": "autonomous systems in software engineering",
        "relevant_chunk_ids": [10, 11],
    },
    {
        "query": "challenges of agentic software engineering",
        "relevant_chunk_ids": [97, 98, 99, 100, 101, 102, 103, 104],
    },
    {
        "query": "evaluation methodologies for agentic software engineering",
        "relevant_chunk_ids": [78, 79, 80, 81],
    },
    {
        "query": "multi-agent systems in software engineering",
        "relevant_chunk_ids": [72, 73, 74],
    },
    {
        "query": "external tools and environments for software engineering agents",
        "relevant_chunk_ids": [75, 76, 77],
    },
    {
        "query": "LLM integration in agentic software engineering",
        "relevant_chunk_ids": [50, 51, 52],
    },
    {
        "query": "goal-oriented and behavior-driven agents",
        "relevant_chunk_ids": [44, 45, 46],
    },
    {
        "query": "software maintenance agents",
        "relevant_chunk_ids": [91, 92],
    },
    {
        "query": "human-agent collaboration and trust",
        "relevant_chunk_ids": [102, 103, 104],
    },
    {
        "query": "planning and reasoning for software engineering agents",
        "relevant_chunk_ids": [82, 83, 84],
    },
]