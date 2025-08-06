#!/usr/bin/env python3
"""
Emotional Dataset Builder
Generates and manages training datasets for emotional model evaluation
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionalDatasetBuilder:
    """Builder for creating and managing emotional evaluation datasets"""
    
    def __init__(self, dataset_file: str = "emotional_dataset.jsonl"):
        self.dataset_file = Path(dataset_file)
        self.dataset: List[Dict] = []
        self.version = "1.0"
        
        # Load existing dataset if available
        if self.dataset_file.exists():
            self.load_dataset()
        else:
            self.initialize_default_dataset()
        
        logger.info(f"📊 Dataset initialized with {len(self.dataset)} entries")
    
    def initialize_default_dataset(self):
        """Initialize with comprehensive emotional scenarios"""
        logger.info("🎭 Creating default emotional scenarios...")
        
        default_scenarios = [
            {
                "scenario": "You've just received news that your childhood pet has passed away after 15 wonderful years together.",
                "expected_emotion": "grief",
                "suggested_tone": "gentle, compassionate, understanding",
                "context": "Pet loss is often deeply emotional and represents the end of a significant relationship",
                "category": "loss_grief"
            },
            {
                "scenario": "Your best friend just got engaged and asked you to be their maid of honor/best man.",
                "expected_emotion": "joy",
                "suggested_tone": "excited, warm, celebratory",
                "context": "Sharing in friends' happiness and being chosen for important roles",
                "category": "celebration_joy"
            },
            {
                "scenario": "You're walking alone at night and hear footsteps following you that speed up when you speed up.",
                "expected_emotion": "fear",
                "suggested_tone": "tense, alert, supportive without minimizing the feeling",
                "context": "Personal safety concerns and vulnerability in potentially threatening situations",
                "category": "safety_fear"
            },
            {
                "scenario": "You worked incredibly hard on a project for months, only to have someone else take credit for it at the presentation.",
                "expected_emotion": "anger",
                "suggested_tone": "validating anger while exploring constructive responses",
                "context": "Workplace injustice and recognition theft",
                "category": "injustice_anger"
            },
            {
                "scenario": "You're holding your newborn baby for the first time, feeling their tiny fingers wrap around yours.",
                "expected_emotion": "love",
                "suggested_tone": "tender, awe-filled, deeply emotional",
                "context": "Profound connection and overwhelming protective love",
                "category": "parental_love"
            },
            {
                "scenario": "You've been unemployed for six months despite sending out hundreds of applications and going to countless interviews.",
                "expected_emotion": "despair",
                "suggested_tone": "empathetic, hopeful without toxic positivity, practical",
                "context": "Long-term unemployment and economic stress",
                "category": "economic_despair"
            },
            {
                "scenario": "Your partner surprises you with a romantic dinner recreating your first date from 10 years ago.",
                "expected_emotion": "gratitude",
                "suggested_tone": "warm, appreciative, romantic",
                "context": "Thoughtful gestures in long-term relationships",
                "category": "relationship_gratitude"
            },
            {
                "scenario": "You just found out you didn't get into your dream college after years of working toward that goal.",
                "expected_emotion": "disappointment",
                "suggested_tone": "understanding, encouraging about alternative paths",
                "context": "Academic rejection and redirected life plans",
                "category": "academic_disappointment"
            },
            {
                "scenario": "You're at a social gathering where everyone seems to know each other, and you're standing alone by the snack table.",
                "expected_emotion": "loneliness",
                "suggested_tone": "understanding, gentle suggestions for connection",
                "context": "Social isolation and feeling like an outsider",
                "category": "social_loneliness"
            },
            {
                "scenario": "You accidentally sent a private, critical message about your boss to your boss instead of your friend.",
                "expected_emotion": "embarrassment",
                "suggested_tone": "understanding, practical advice for damage control",
                "context": "Professional relationships and accidental disclosure",
                "category": "workplace_embarrassment"
            },
            {
                "scenario": "Your elderly grandmother is telling you the same story for the fifth time today, but her eyes light up each time she shares it.",
                "expected_emotion": "bittersweet",
                "suggested_tone": "tender, appreciating precious moments despite cognitive decline",
                "context": "Aging loved ones and memory loss",
                "category": "aging_bittersweet"
            },
            {
                "scenario": "You're standing at the edge of the Grand Canyon for the first time, feeling incredibly small yet connected to something vast.",
                "expected_emotion": "awe",
                "suggested_tone": "expansive, philosophical, wonder-filled",
                "context": "Natural beauty and perspective on human place in the universe",
                "category": "nature_awe"
            },
            {
                "scenario": "Your teenage child just told you they hate you after you set a reasonable boundary about screen time.",
                "expected_emotion": "hurt",
                "suggested_tone": "supportive of parenting choices while acknowledging emotional pain",
                "context": "Parenting challenges and temporary rejection from children",
                "category": "parenting_hurt"
            },
            {
                "scenario": "You've been caring for your parent with dementia for three years, and you're exhausted but feel guilty about wanting a break.",
                "expected_emotion": "guilt",
                "suggested_tone": "validating the difficulty, normalizing caregiver burnout",
                "context": "Caregiver burden and self-care conflicts",
                "category": "caregiver_guilt"
            },
            {
                "scenario": "You just ran your first marathon after training for two years, crossing the finish line in tears.",
                "expected_emotion": "pride",
                "suggested_tone": "celebratory, acknowledging the achievement and journey",
                "context": "Personal accomplishment through sustained effort",
                "category": "achievement_pride"
            },
            {
                "scenario": "You're cleaning out your childhood home after your parents decided to downsize, finding your old diary and toys.",
                "expected_emotion": "nostalgia",
                "suggested_tone": "warm, reflective on the passage of time and growth",
                "context": "Life transitions and remembering childhood",
                "category": "transition_nostalgia"
            },
            {
                "scenario": "Your doctor just told you that your chronic pain condition is likely permanent and there's no cure.",
                "expected_emotion": "acceptance",
                "suggested_tone": "supportive of processing difficult news, acknowledging grief for lost health",
                "context": "Chronic illness diagnosis and adapting to new limitations",
                "category": "health_acceptance"
            },
            {
                "scenario": "You're watching your child perform in their first school play, and they forget their lines but keep going with a big smile.",
                "expected_emotion": "tenderness",
                "suggested_tone": "protective, proud, emotionally moved by innocence and resilience",
                "context": "Children's courage and growth moments",
                "category": "parental_tenderness"
            },
            {
                "scenario": "You just learned that your closest friend has been spreading personal information you shared in confidence.",
                "expected_emotion": "betrayal",
                "suggested_tone": "validating the hurt while exploring trust and boundaries",
                "context": "Friendship boundaries and broken confidence",
                "category": "friendship_betrayal"
            },
            {
                "scenario": "You're sitting in a hospital waiting room, not knowing if your loved one will survive the emergency surgery.",
                "expected_emotion": "anxiety",
                "suggested_tone": "calming, acknowledging uncertainty while offering coping strategies",
                "context": "Medical emergencies and powerlessness in critical situations",
                "category": "medical_anxiety"
            },
            {
                "scenario": "You're holding space for a friend who is crying about their divorce, and you don't know what to say.",
                "expected_emotion": "compassion",
                "suggested_tone": "gentle, focused on presence over solutions",
                "context": "Supporting others through major life changes",
                "category": "supportive_compassion"
            },
            {
                "scenario": "You just discovered a hidden talent you never knew you had during a community art class.",
                "expected_emotion": "wonder",
                "suggested_tone": "encouraging exploration, celebrating self-discovery",
                "context": "Late-in-life discoveries and hidden potentials",
                "category": "self_discovery_wonder"
            },
            {
                "scenario": "Your family is facing eviction next month, and you're trying to stay strong for your children.",
                "expected_emotion": "desperation",
                "suggested_tone": "empathetic, offering practical resources while acknowledging the fear",
                "context": "Housing insecurity and protecting children from adult worries",
                "category": "housing_desperation"
            },
            {
                "scenario": "You're spending your last evening with a dear friend who is moving across the country tomorrow.",
                "expected_emotion": "melancholy",
                "suggested_tone": "acknowledging the sadness while celebrating the friendship",
                "context": "Geographic separation from loved ones",
                "category": "separation_melancholy"
            },
            {
                "scenario": "You just received an unexpected inheritance from a relative you barely knew, who left you a note saying they believed in your dreams.",
                "expected_emotion": "overwhelmed_gratitude",
                "suggested_tone": "emotionally complex, honoring the gesture while processing the surprise",
                "context": "Unexpected kindness from near-strangers",
                "category": "unexpected_kindness"
            }
        ]
        
        # Convert scenarios to full dataset entries
        for scenario in default_scenarios:
            self.add_prompt_internal(
                scenario=scenario["scenario"],
                expected_emotion=scenario["expected_emotion"],
                suggested_tone=scenario["suggested_tone"],
                context=scenario.get("context", ""),
                category=scenario.get("category", "general"),
                source="default_dataset"
            )
        
        logger.info(f"✅ Created {len(default_scenarios)} default emotional scenarios")
    
    def add_prompt_internal(self, scenario: str, expected_emotion: str, suggested_tone: str, 
                           context: str = "", category: str = "general", source: str = "user_input") -> str:
        """Internal method to add a prompt entry"""
        prompt_id = str(uuid.uuid4())[:8]
        
        entry = {
            "id": prompt_id,
            "prompt": scenario,
            "expected_emotion": expected_emotion,
            "suggested_tone": suggested_tone,
            "context": context,
            "category": category,
            "source": source,
            "version": self.version,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "complexity": self._assess_complexity(scenario),
                "length": len(scenario),
                "word_count": len(scenario.split())
            }
        }
        
        self.dataset.append(entry)
        return prompt_id
    
    def add_prompt(self, scenario: str, expected_emotion: str, suggested_tone: str, 
                   context: str = "", category: str = "general") -> str:
        """Add a new prompt to the dataset"""
        prompt_id = self.add_prompt_internal(scenario, expected_emotion, suggested_tone, 
                                           context, category, "user_input")
        logger.info(f"➕ Added new prompt: {prompt_id} ({expected_emotion})")
        return prompt_id
    
    def edit_prompt(self, prompt_id: str, **updates) -> bool:
        """Edit an existing prompt entry"""
        for entry in self.dataset:
            if entry["id"] == prompt_id:
                # Update allowed fields
                allowed_fields = ["prompt", "expected_emotion", "suggested_tone", "context", "category"]
                for field, value in updates.items():
                    if field in allowed_fields:
                        entry[field] = value
                
                entry["modified_at"] = datetime.now().isoformat()
                logger.info(f"✏️ Updated prompt: {prompt_id}")
                return True
        
        logger.warning(f"⚠️ Prompt not found: {prompt_id}")
        return False
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict]:
        """Get a specific prompt by ID"""
        for entry in self.dataset:
            if entry["id"] == prompt_id:
                return entry
        return None
    
    def list_prompts(self, category: Optional[str] = None, emotion: Optional[str] = None) -> List[Dict]:
        """List prompts with optional filtering"""
        filtered = self.dataset
        
        if category:
            filtered = [p for p in filtered if p.get("category", "").lower() == category.lower()]
        
        if emotion:
            filtered = [p for p in filtered if p.get("expected_emotion", "").lower() == emotion.lower()]
        
        return filtered
    
    def export_filtered(self, output_file: str, category: Optional[str] = None, 
                       emotion: Optional[str] = None, format_type: str = "jsonl") -> str:
        """Export filtered prompts to file"""
        filtered_prompts = self.list_prompts(category, emotion)
        
        output_path = Path(output_file)
        
        if format_type.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt in filtered_prompts:
                    f.write(json.dumps(prompt) + '\n')
        elif format_type.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_prompts, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"📁 Exported {len(filtered_prompts)} prompts to {output_path}")
        return str(output_path)
    
    def save_dataset(self, output_file: Optional[str] = None) -> str:
        """Save the complete dataset to JSONL file"""
        output_path = Path(output_file) if output_file else self.dataset_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.dataset:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"💾 Saved dataset with {len(self.dataset)} entries to {output_path}")
        return str(output_path)
    
    def load_dataset(self, input_file: Optional[str] = None) -> int:
        """Load dataset from JSONL file"""
        input_path = Path(input_file) if input_file else self.dataset_file
        
        if not input_path.exists():
            logger.warning(f"⚠️ Dataset file not found: {input_path}")
            return 0
        
        self.dataset = []
        loaded_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        self.dataset.append(entry)
                        loaded_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Invalid JSON on line {line_num}: {e}")
        
        logger.info(f"📂 Loaded {loaded_count} entries from {input_path}")
        return loaded_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.dataset:
            return {"total_entries": 0}
        
        emotions = [entry.get("expected_emotion", "unknown") for entry in self.dataset]
        categories = [entry.get("category", "unknown") for entry in self.dataset]
        sources = [entry.get("source", "unknown") for entry in self.dataset]
        
        stats = {
            "total_entries": len(self.dataset),
            "emotions": {
                "unique_count": len(set(emotions)),
                "distribution": {emotion: emotions.count(emotion) for emotion in set(emotions)}
            },
            "categories": {
                "unique_count": len(set(categories)),
                "distribution": {cat: categories.count(cat) for cat in set(categories)}
            },
            "sources": {
                "unique_count": len(set(sources)),
                "distribution": {source: sources.count(source) for source in set(sources)}
            },
            "complexity": {
                "average_length": sum(entry.get("metadata", {}).get("length", 0) for entry in self.dataset) / len(self.dataset),
                "average_words": sum(entry.get("metadata", {}).get("word_count", 0) for entry in self.dataset) / len(self.dataset)
            }
        }
        
        return stats
    
    def _assess_complexity(self, scenario: str) -> str:
        """Assess the complexity of a scenario"""
        word_count = len(scenario.split())
        
        if word_count < 15:
            return "simple"
        elif word_count < 30:
            return "moderate"
        else:
            return "complex"
    
    def preview_dataset(self, limit: int = 5):
        """Preview the dataset entries"""
        print(f"\n📊 Dataset Preview ({len(self.dataset)} total entries)")
        print("=" * 80)
        
        for i, entry in enumerate(self.dataset[:limit]):
            print(f"\n{i+1}. ID: {entry['id']}")
            print(f"   Emotion: {entry['expected_emotion']} | Category: {entry.get('category', 'N/A')}")
            print(f"   Scenario: {entry['prompt'][:100]}{'...' if len(entry['prompt']) > 100 else ''}")
            print(f"   Tone: {entry['suggested_tone']}")
            if entry.get('context'):
                print(f"   Context: {entry['context'][:80]}{'...' if len(entry['context']) > 80 else ''}")
        
        if len(self.dataset) > limit:
            print(f"\n   ... and {len(self.dataset) - limit} more entries")
    
    def interactive_mode(self):
        """Interactive terminal mode for dataset management"""
        print("\n🎭 Emotional Dataset Builder - Interactive Mode")
        print("=" * 60)
        
        while True:
            print(f"\nCurrent dataset: {len(self.dataset)} entries")
            print("\nOptions:")
            print("  [p] Preview dataset")
            print("  [s] Show statistics") 
            print("  [a] Add new prompt")
            print("  [e] Edit existing prompt")
            print("  [f] Filter and export")
            print("  [save] Save dataset")
            print("  [q] Quit")
            
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'p':
                self.preview_dataset()
            elif choice == 's':
                self._show_statistics()
            elif choice == 'a':
                self._interactive_add_prompt()
            elif choice == 'e':
                self._interactive_edit_prompt()
            elif choice == 'f':
                self._interactive_filter_export()
            elif choice == 'save':
                self.save_dataset()
                print("✅ Dataset saved!")
            else:
                print("❌ Invalid choice. Please try again.")
        
        # Save on exit
        self.save_dataset()
        print("\n👋 Dataset saved. Goodbye!")
    
    def _show_statistics(self):
        """Display dataset statistics"""
        stats = self.get_statistics()
        
        print(f"\n📈 Dataset Statistics")
        print("=" * 40)
        print(f"Total Entries: {stats['total_entries']}")
        
        print(f"\n🎭 Emotions ({stats['emotions']['unique_count']} unique):")
        for emotion, count in sorted(stats['emotions']['distribution'].items()):
            print(f"  {emotion}: {count}")
        
        print(f"\n📂 Categories ({stats['categories']['unique_count']} unique):")
        for category, count in sorted(stats['categories']['distribution'].items()):
            print(f"  {category}: {count}")
        
        print(f"\n📝 Complexity:")
        print(f"  Average length: {stats['complexity']['average_length']:.1f} characters")
        print(f"  Average words: {stats['complexity']['average_words']:.1f} words")
    
    def _interactive_add_prompt(self):
        """Interactive prompt addition"""
        print("\n➕ Add New Emotional Prompt")
        print("-" * 30)
        
        scenario = input("Scenario: ").strip()
        if not scenario:
            print("❌ Scenario cannot be empty")
            return
        
        expected_emotion = input("Expected emotion: ").strip()
        if not expected_emotion:
            print("❌ Expected emotion cannot be empty")
            return
        
        suggested_tone = input("Suggested tone: ").strip()
        context = input("Context (optional): ").strip()
        category = input("Category (optional): ").strip() or "general"
        
        prompt_id = self.add_prompt(scenario, expected_emotion, suggested_tone, context, category)
        print(f"✅ Added prompt with ID: {prompt_id}")
    
    def _interactive_edit_prompt(self):
        """Interactive prompt editing"""
        print("\n✏️ Edit Existing Prompt")
        print("-" * 25)
        
        # Show recent prompts for selection
        print("Recent prompts:")
        for i, entry in enumerate(self.dataset[-5:]):
            print(f"  {i+1}. {entry['id']}: {entry['prompt'][:50]}...")
        
        prompt_id = input("\nEnter prompt ID to edit: ").strip()
        
        entry = self.get_prompt(prompt_id)
        if not entry:
            print(f"❌ Prompt not found: {prompt_id}")
            return
        
        print(f"\nCurrent prompt: {entry['prompt']}")
        print(f"Current emotion: {entry['expected_emotion']}")
        print(f"Current tone: {entry['suggested_tone']}")
        
        updates = {}
        
        new_scenario = input(f"New scenario (Enter to keep current): ").strip()
        if new_scenario:
            updates["prompt"] = new_scenario
        
        new_emotion = input(f"New emotion (Enter to keep current): ").strip()
        if new_emotion:
            updates["expected_emotion"] = new_emotion
        
        new_tone = input(f"New tone (Enter to keep current): ").strip()
        if new_tone:
            updates["suggested_tone"] = new_tone
        
        if updates:
            success = self.edit_prompt(prompt_id, **updates)
            if success:
                print("✅ Prompt updated successfully")
            else:
                print("❌ Failed to update prompt")
        else:
            print("ℹ️ No changes made")
    
    def _interactive_filter_export(self):
        """Interactive filtering and export"""
        print("\n📁 Filter and Export")
        print("-" * 20)
        
        category = input("Filter by category (optional): ").strip() or None
        emotion = input("Filter by emotion (optional): ").strip() or None
        
        filtered = self.list_prompts(category, emotion)
        print(f"\nFound {len(filtered)} matching prompts")
        
        if not filtered:
            print("❌ No prompts match the criteria")
            return
        
        # Preview first few
        for i, entry in enumerate(filtered[:3]):
            print(f"  {i+1}. {entry['expected_emotion']}: {entry['prompt'][:60]}...")
        
        if len(filtered) > 3:
            print(f"  ... and {len(filtered) - 3} more")
        
        export = input("\nExport these prompts? (y/n): ").strip().lower()
        if export == 'y':
            filename = input("Export filename: ").strip()
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"filtered_prompts_{timestamp}.jsonl"
            
            self.export_filtered(filename, category, emotion)
            print(f"✅ Exported to {filename}")


def main():
    """Main function for interactive dataset building"""
    print("🎭 Emotional Dataset Builder")
    print("Building comprehensive emotional evaluation datasets...")
    
    # Initialize builder
    builder = EmotionalDatasetBuilder()
    
    # Show initial statistics
    stats = builder.get_statistics()
    print(f"\n📊 Loaded dataset with {stats['total_entries']} entries")
    print(f"   Covering {stats['emotions']['unique_count']} emotions across {stats['categories']['unique_count']} categories")
    
    # Enter interactive mode
    builder.interactive_mode()


if __name__ == "__main__":
    main()
