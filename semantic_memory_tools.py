import json
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
        context["category"] = input_data.category.lower()
    if input_data.location:
        context["location"] = input_data.location.lower()
    if input_data.participants:
        context["participants"] = json.dumps(input_data.participants)
    if input_data.importance:
        context["importance"] = input_data.importance
    if input_data.tags:
        context["tags"] = json.dumps([tag.lower() for tag in input_data.tags])

    # Store the memory
    memory_id = memory.store(input_data.content, context)

    return f"Successfully stored memory with ID: {memory_id}"


def get_memory_entry_formatted(memory_entry) -> str:
    output = f""
    output += f"Category:\n{memory_entry['metadata'].get('category', 'Unknown')}\n\n"
    output += f"Location:\n{memory_entry['metadata'].get('location', 'Unknown')}\n\n"
    output += f"Content:\n{memory_entry['content']}\n\n"
    output += f"Participants:\n{', '.join(json.loads(memory_entry['metadata'].get('participants', '[]')))}\n\n"
    output += f"Importance:\n{memory_entry['metadata'].get('importance', 'Unknown')}\n\n"
    output += f"Tags:\n{', '.join(json.loads(memory_entry['metadata'].get('tags', '[]')))}\n"
    return output


def get_memory_entries_formatted(memory_entries) -> str:
    output = "--- Memories ---\n"

    for idx, memory_entry in enumerate(memory_entries):
        output += f"# Memory {idx + 1}\n{get_memory_entry_formatted(memory_entry)}\n"

    output += "--- Memories End ---\n"
    return output


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

    # Build context filter conditions
    conditions = []
    if input_data.category:
        conditions.append({"category": input_data.category.lower()})
    if input_data.location:
        conditions.append({"location": input_data.location.lower()})
    if input_data.min_importance:
        conditions.append({"importance": {"$gte": input_data.min_importance}})
    if input_data.tags:
        conditions.append({"tags": {"$in": [tag.lower() for tag in input_data.tags]}})

    # Combine conditions with $and operator if there are multiple conditions
    context_filter = (
        {"$and": conditions} if len(conditions) > 1
        else conditions[0] if conditions
        else None
    )

    # Perform recall
    results = memory.recall(
        query=input_data.query,
        n_results=input_data.n_results,
        context_filter=context_filter if len(context_filter) > 0 else None
    )
    output = get_memory_entries_formatted(results)
    return output


class AnalyzeMemoryPatternsInput(BaseModel):
    """Input for analyzing memory patterns."""
    timeframe: Optional[str] = Field(None, description="Timeframe to analyze (e.g., '1h', '24h', '7d')")
    category: Optional[str] = Field(None, description="Filter by category")
    location: Optional[str] = Field(None, description="Filter by location")
    min_similarity: Optional[float] = Field(0.7, description="Minimum similarity threshold", ge=0.0, le=1.0)


class SummarizeMemoriesInput(BaseModel):
    """Input for summarizing memories."""
    category: Optional[str] = Field(None, description="Filter by category")
    location: Optional[str] = Field(None, description="Filter by location")
    timeframe: Optional[str] = Field(None, description="Timeframe to summarize (e.g., '1h', '24h', '7d')")
    importance_threshold: Optional[int] = Field(None, description="Minimum importance level to include", ge=1, le=10)


def get_memory_count() -> str:
    global memory
    stats = memory.get_stats()
    return f"Total memories: {sum(stats.values())}\n"


# Create function tools
store_memory_tool = FunctionTool(store_memory)
recall_memories_tool = FunctionTool(recall_memories)

# Example usage
if __name__ == "__main__":
    # Test storing a memory
    store_input = StoreMemoryInput(
        content="The party encountered a group of goblins in the Dark Forest",
        category="Combat",
        location="Dark Forest",
        participants=["Thorin", "Elara", "Redrick"],
        importance=7,
        tags=["Combat", "Goblins", "Forest"]
    )
    result = store_memory(store_input)
    print(result)

    # Test recalling memories
    recall_input = RecallMemoryInput(
        query="What happened in the forest?",
        category="Combat",
        location="Dark Forest",
        min_importance=5
    )
    memories = recall_memories(recall_input)
    print(get_memory_entries_formatted(memories))
