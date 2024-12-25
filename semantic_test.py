import json
import random
from datetime import datetime, timedelta
from typing import Dict

from semantic_memory import SemanticMemory


class RPGMemoryTest:
    def __init__(self, persist_directory: str = "./rpg_memory"):
        """Initialize the RPG memory test system"""
        self.memory = SemanticMemory(persist_directory)
        self.current_time = datetime.now()

        # Track game state
        self.party_members = [
            "Thorin the Dwarf Fighter",
            "Elara the Elf Mage",
            "Redrick the Human Rogue"
        ]
        self.current_location = "Town of Riverbrook"
        self.active_quests = []

    def simulate_game_session(self, num_events: int = 10):
        """Simulate a game session with multiple events"""
        for _ in range(num_events):
            event = self._generate_random_event()
            self._process_event(event)
            self.current_time += timedelta(minutes=random.randint(10, 60))

    def _generate_random_event(self) -> Dict:
        """Generate a random game event"""
        event_types = [
            self._generate_combat_event,
            self._generate_npc_interaction,
            self._generate_quest_event,
            self._generate_exploration_event,
            self._generate_character_development
        ]
        return random.choice(event_types)()

    def _generate_combat_event(self) -> Dict:
        """Generate a combat-related event"""
        enemies = [
            "goblin raiders", "wolf pack", "bandit group",
            "undead warriors", "giant spider"
        ]
        outcomes = ["victory", "retreat", "near defeat", "total victory"]

        event = {
            "type": "combat",
            "enemy": random.choice(enemies),
            "outcome": random.choice(outcomes),
            "participants": self.party_members,
            "location": self.current_location
        }

        return event

    def _generate_npc_interaction(self) -> Dict:
        """Generate an NPC interaction event"""
        npcs = [
            {"name": "Elder Miriam", "role": "town elder"},
            {"name": "Blacksmith Jorge", "role": "craftsman"},
            {"name": "Mysterious Stranger", "role": "unknown"},
            {"name": "Merchant Kira", "role": "trader"}
        ]

        interaction_types = ["trade", "information", "quest", "relationship"]
        npc = random.choice(npcs)

        event = {
            "type": "npc_interaction",
            "npc": npc["name"],
            "npc_role": npc["role"],
            "interaction_type": random.choice(interaction_types),
            "location": self.current_location
        }

        return event

    def _generate_quest_event(self) -> Dict:
        """Generate a quest-related event"""
        quest_types = ["rescue", "retrieve", "investigate", "protect"]
        quest_states = ["started", "progress", "completed", "failed"]

        event = {
            "type": "quest",
            "quest_type": random.choice(quest_types),
            "state": random.choice(quest_states),
            "location": self.current_location
        }

        return event

    def _generate_exploration_event(self) -> Dict:
        """Generate an exploration event"""
        locations = [
            "Ancient Ruins", "Dark Forest", "Underground Cavern",
            "Mountain Pass", "Abandoned Village"
        ]
        discoveries = [
            "hidden treasure", "mysterious artifact",
            "ancient inscription", "rare herb patch",
            "secret passage"
        ]

        new_location = random.choice(locations)
        event = {
            "type": "exploration",
            "previous_location": self.current_location,
            "new_location": new_location,
            "discovery": random.choice(discoveries)
        }

        self.current_location = new_location
        return event

    def _generate_character_development(self) -> Dict:
        """Generate a character development event"""
        character = random.choice(self.party_members)
        developments = [
            "learned new skill", "found magical item",
            "gained level", "character background revelation",
            "personal quest progress"
        ]

        event = {
            "type": "character_development",
            "character": character,
            "development": random.choice(developments),
            "location": self.current_location
        }

        return event

    def _process_event(self, event: Dict):
        """Process and store a game event in memory"""
        # Convert event to narrative description
        narrative = self._event_to_narrative(event)

        # Store in memory with context
        self.memory.store(
            content=narrative,
            context={
                "event_type": event["type"],
                "location": event.get("location", self.current_location),
                "timestamp": self.current_time.isoformat(),
                "participants": json.dumps(event.get("participants", []))
            }
        )

    def _event_to_narrative(self, event: Dict) -> str:
        """Convert event dictionary to narrative description"""
        if event["type"] == "combat":
            return (f"The party encountered {event['enemy']} near {event['location']}. "
                    f"The battle ended in {event['outcome']}. "
                    f"Participants: {', '.join(event['participants'])}.")

        elif event["type"] == "npc_interaction":
            return (f"The party met with {event['npc']} ({event['npc_role']}) "
                    f"in {event['location']} for {event['interaction_type']}.")

        elif event["type"] == "quest":
            return (f"Quest {event['quest_type']} {event['state']} "
                    f"at {event['location']}.")

        elif event["type"] == "exploration":
            return (f"The party traveled from {event['previous_location']} "
                    f"to {event['new_location']} and discovered {event['discovery']}.")

        elif event["type"] == "character_development":
            return (f"{event['character']} {event['development']} "
                    f"while at {event['location']}.")

        return "Unknown event occurred."

    def query_game_history(self, query: str, n_results: int = 5,
                           event_type: str = None, location: str = None):
        """Query the game history with optional filters"""
        context_filter = {}
        if event_type:
            context_filter["event_type"] = event_type
        if location:
            context_filter["location"] = location

        results = self.memory.recall(
            query=query,
            n_results=n_results,
            context_filter=context_filter
        )

        return results


# Example usage
if __name__ == "__main__":
    # Initialize and run test scenario
    rpg_test = RPGMemoryTest("./rpg_memory_test")

    # Simulate a game session
    rpg_test.simulate_game_session(num_events=350)

    # Query examples
    print("\nQuerying for combat events:")
    combat_results = rpg_test.query_game_history(
        "What battles happened?",
        event_type="combat"
    )
    for result in combat_results:
        print(f"Memory: {result['content']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print("---")

    print("\nQuerying for character development:")
    character_results = rpg_test.query_game_history(
        "What happened to Thorin?",
        n_results=3
    )
    for result in character_results:
        print(f"Memory: {result['content']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print("---")

    print("\nQuerying location-specific events:")
    location_results = rpg_test.query_game_history(
        "What happened in the Ancient Ruins?",
        location="Ancient Ruins"
    )
    for result in location_results:
        print(f"Memory: {result['content']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print("---")
