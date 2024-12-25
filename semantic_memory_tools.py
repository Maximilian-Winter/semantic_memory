from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from semantic_memory import SemanticMemory
from ToolAgents import FunctionTool

memory = SemanticMemory("./game_master_memory")


class StoreMemoryInput(BaseModel):
    """Input for storing a new memory."""
    content: str = Field(..., description="The content of the memory to store")
    category: Optional[str] = Field(None,
                                    description="Category of the memory (e.g., 'combat', 'npc_interaction', 'quest')")
    location: Optional[str] = Field(None, description="Location where the memory took place")
    participants: Optional[List[str]] = Field(None, description="List of participants involved in the memory")
    importance: Optional[int] = Field(None, description="Importance level of the memory (1-10)", ge=1, le=10)
    tags: Optional[List[str]] = Field(None, description="List of tags to associate with the memory")


def store_memory(input_data: StoreMemoryInput) -> str:
    """
    Store a new memory in the semantic memory system.

    Args:
        input_data: The input containing memory content and metadata
    Returns:
        A message confirming the memory was stored
    """
    global memory
    # Prepare context metadata
    context = {
        "timestamp": datetime.now().isoformat(),
        "type": "memory"
    }

    # Add optional fields to context if they exist
    if input_data.category:
        context["category"] = input_data.category
    if input_data.location:
        context["location"] = input_data.location
    if input_data.participants:
        context["participants"] = input_data.participants
    if input_data.importance:
        context["importance"] = input_data.importance
    if input_data.tags:
        context["tags"] = input_data.tags

    # Store the memory
    memory_id = memory.store(input_data.content, context)

    return f"Successfully stored memory with ID: {memory_id}"


class RecallMemoryInput(BaseModel):
    """Input for recalling memories."""
    query: str = Field(..., description="The query to search for in memories")
    n_results: Optional[int] = Field(5, description="Number of results to return")
    category: Optional[str] = Field(None, description="Filter by category")
    location: Optional[str] = Field(None, description="Filter by location")
    min_importance: Optional[int] = Field(None, description="Minimum importance level", ge=1, le=10)
    tags: Optional[List[str]] = Field(None, description="Filter by tags (any match)")


def recall_memories(input_data: RecallMemoryInput) -> List[Dict[str, Any]]:
    """
    Recall memories based on a query and optional filters.

    Args:
        input_data: The input containing search parameters
    Returns:
        A list of matching memories with their metadata
    """
    global memory

    # Build context filter
    context_filter = {}
    if input_data.category:
        context_filter["category"] = input_data.category
    if input_data.location:
        context_filter["location"] = input_data.location
    if input_data.min_importance:
        context_filter["importance"] = {"$gte": input_data.min_importance}
    if input_data.tags:
        context_filter["tags"] = {"$in": input_data.tags}

    # Perform recall
    results = memory.recall(
        query=input_data.query,
        n_results=input_data.n_results,
        context_filter=context_filter if context_filter else None
    )

    return results


class AnalyzeMemoryPatternsInput(BaseModel):
    """Input for analyzing memory patterns."""
    timeframe: Optional[str] = Field(None, description="Timeframe to analyze (e.g., '1h', '24h', '7d')")
    category: Optional[str] = Field(None, description="Filter by category")
    location: Optional[str] = Field(None, description="Filter by location")
    min_similarity: Optional[float] = Field(0.7, description="Minimum similarity threshold", ge=0.0, le=1.0)


def analyze_memory_patterns(input_data: AnalyzeMemoryPatternsInput) -> Dict[str, Any]:
    """
    Analyze patterns in stored memories.

    Args:
        input_data: The input containing analysis parameters
    Returns:
        Analysis results including patterns and statistics
    """
    global memory

    # Get memory statistics
    stats = memory.get_stats()

    # For now, return basic statistics
    # This could be expanded to include more sophisticated pattern analysis
    return {
        "memory_counts": stats,
        "analysis_parameters": {
            "timeframe": input_data.timeframe,
            "category": input_data.category,
            "location": input_data.location,
            "min_similarity": input_data.min_similarity
        }
    }


class SummarizeMemoriesInput(BaseModel):
    """Input for summarizing memories."""
    category: Optional[str] = Field(None, description="Filter by category")
    location: Optional[str] = Field(None, description="Filter by location")
    timeframe: Optional[str] = Field(None, description="Timeframe to summarize (e.g., '1h', '24h', '7d')")
    importance_threshold: Optional[int] = Field(None, description="Minimum importance level to include", ge=1, le=10)


def summarize_memories(input_data: SummarizeMemoriesInput) -> str:
    """
    Generate a summary of stored memories based on filters.

    Args:
        input_data: The input containing summarization parameters
    Returns:
        A textual summary of the memories
    """
    global memory

    # Build context filter
    context_filter = {}
    if input_data.category:
        context_filter["category"] = input_data.category
    if input_data.location:
        context_filter["location"] = input_data.location
    if input_data.importance_threshold:
        context_filter["importance"] = {"$gte": input_data.importance_threshold}

    # For now, return a basic summary
    # This could be expanded to generate more detailed summaries
    stats = memory.get_stats()
    return (
        f"Memory Summary:\n"
        f"Total memories: {sum(stats.values())}\n"
        f"Immediate memories: {stats['immediate_count']}\n"
        f"Working memories: {stats['working_count']}\n"
        f"Long-term memories: {stats['long_term_count']}\n"
        f"Filters applied: {', '.join(f'{k}={v}' for k, v in input_data.dict(exclude_none=True).items())}"
    )


# Create function tools
store_memory_tool = FunctionTool(store_memory)
recall_memories_tool = FunctionTool(recall_memories)
analyze_patterns_tool = FunctionTool(analyze_memory_patterns)
summarize_memories_tool = FunctionTool(summarize_memories)

# Example usage
if __name__ == "__main__":
    # Test storing a memory
    store_input = StoreMemoryInput(
        content="The party encountered a group of goblins in the Dark Forest",
        category="combat",
        location="Dark Forest",
        participants=["Thorin", "Elara", "Redrick"],
        importance=7,
        tags=["combat", "goblins", "forest"]
    )
    result = store_memory(store_input)
    print(result)

    # Test recalling memories
    recall_input = RecallMemoryInput(
        query="What happened in the forest?",
        category="combat",
        location="Dark Forest",
        min_importance=5
    )
    memories = recall_memories(recall_input)
    for memory in memories:
        print(f"\nMemory: {memory['content']}")
        print(f"Similarity: {memory['similarity']:.2f}")
        print(f"Type: {memory['memory_type']}")
