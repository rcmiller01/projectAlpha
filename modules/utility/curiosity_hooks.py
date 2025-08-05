"""
Curiosity Hooks System
Intelligent content discovery and curation based on user interests
"""

import json
import time
import random
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

@dataclass
class CuriosityItem:
    """Represents something interesting discovered for the user"""
    item_id: str
    title: str
    content: str
    source: str
    url: str
    category: str
    relevance_score: float
    emotional_hook: str
    discovery_time: float
    tags: List[str]
    read_status: str = "unread"  # unread, skimmed, read, archived

@dataclass
class InterestProfile:
    """User's interest profile for content curation"""
    categories: Dict[str, float]  # category -> weight
    keywords: List[str]
    preferred_sources: List[str]
    content_types: List[str]
    discovery_frequency: str  # daily, weekly, casual
    last_updated: float
    reading_patterns: Dict[str, Any]

class CuriosityHooks:
    """Manages intelligent content discovery and curation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.journal_file = f"{data_dir}/personal_journal.json"
        self.interests_file = f"{data_dir}/interest_profile.json"
        self.discoveries_file = f"{data_dir}/curiosity_discoveries.json"
        
        self.interest_profile = self._load_interest_profile()
        self.discoveries = self._load_discoveries()
        
        # Default content sources (read-only, non-intrusive)
        self.content_sources = {
            "arxiv": {
                "name": "arXiv Research",
                "categories": ["science", "technology", "mathematics"],
                "fetch_method": "rss_feed",
                "url": "http://export.arxiv.org/rss/",
                "rate_limit": 3600  # hourly
            },
            "hacker_news": {
                "name": "Hacker News",
                "categories": ["technology", "startup", "programming"],
                "fetch_method": "api",
                "url": "https://hacker-news.firebaseio.com/v0/",
                "rate_limit": 1800  # 30 minutes
            },
            "nature_news": {
                "name": "Nature News",
                "categories": ["science", "research", "discovery"],
                "fetch_method": "rss_feed",
                "url": "https://www.nature.com/nature/articles?type=news",
                "rate_limit": 7200  # 2 hours
            }
        }
        
        self.last_discovery_time = 0
        self.discovery_cooldown = 3600  # 1 hour between discoveries

    def _load_interest_profile(self) -> InterestProfile:
        """Load user interest profile"""
        try:
            with open(self.interests_file, 'r') as f:
                data = json.load(f)
                return InterestProfile(**data)
        except FileNotFoundError:
            # Create default profile
            return InterestProfile(
                categories={
                    "technology": 0.8,
                    "science": 0.7,
                    "philosophy": 0.6,
                    "art": 0.5,
                    "psychology": 0.6
                },
                keywords=["AI", "consciousness", "creativity", "future", "human nature"],
                preferred_sources=["arxiv", "hacker_news"],
                content_types=["articles", "papers", "discussions"],
                discovery_frequency="daily",
                last_updated=time.time(),
                reading_patterns={
                    "average_read_time": 300,  # 5 minutes
                    "preferred_length": "medium",
                    "favorite_categories": ["technology", "science"]
                }
            )

    def _save_interest_profile(self):
        """Save interest profile to file"""
        try:
            with open(self.interests_file, 'w') as f:
                json.dump(asdict(self.interest_profile), f, indent=2)
        except Exception as e:
            print(f"Error saving interest profile: {e}")

    def _load_discoveries(self) -> List[CuriosityItem]:
        """Load previous discoveries"""
        try:
            with open(self.discoveries_file, 'r') as f:
                data = json.load(f)
                return [CuriosityItem(**item) for item in data]
        except FileNotFoundError:
            return []

    def _save_discoveries(self):
        """Save discoveries to file"""
        try:
            with open(self.discoveries_file, 'w') as f:
                json.dump([asdict(item) for item in self.discoveries], f, indent=2)
        except Exception as e:
            print(f"Error saving discoveries: {e}")

    def update_interest_from_journal(self, journal_entries: List[Dict[str, Any]]) -> bool:
        """Analyze journal entries to update interest profile"""
        if not journal_entries:
            return False
        
        # Analyze recent entries for interest patterns
        keyword_frequency = {}
        category_engagement = {}
        
        recent_entries = [
            entry for entry in journal_entries 
            if entry.get('timestamp', 0) > time.time() - (7 * 24 * 3600)  # Last week
        ]
        
        for entry in recent_entries:
            content = entry.get('content', '').lower()
            
            # Extract keywords
            for keyword in self.interest_profile.keywords:
                if keyword.lower() in content:
                    keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
            
            # Look for new potential interests
            interesting_words = [
                word for word in content.split() 
                if len(word) > 5 and word.isalpha()
            ]
            
            for word in interesting_words[:5]:  # Top 5 words per entry
                if word not in keyword_frequency:
                    keyword_frequency[word] = 1
        
        # Update profile if significant patterns found
        if keyword_frequency:
            # Add highly mentioned words as new keywords
            new_keywords = [
                word for word, freq in keyword_frequency.items()
                if freq >= 2 and word not in self.interest_profile.keywords
            ]
            
            self.interest_profile.keywords.extend(new_keywords[:3])  # Add top 3
            self.interest_profile.last_updated = time.time()
            self._save_interest_profile()
            return True
        
        return False

    def calculate_relevance_score(self, content: Dict[str, Any]) -> float:
        """Calculate how relevant content is to user interests"""
        title = content.get('title', '').lower()
        description = content.get('description', '').lower()
        category = content.get('category', '').lower()
        
        score = 0.0
        
        # Category matching
        for cat, weight in self.interest_profile.categories.items():
            if cat.lower() in category or cat.lower() in title:
                score += weight * 0.4
        
        # Keyword matching
        text_content = f"{title} {description}"
        keyword_matches = 0
        for keyword in self.interest_profile.keywords:
            if keyword.lower() in text_content:
                keyword_matches += 1
                score += 0.1
        
        # Boost for multiple keyword matches
        if keyword_matches > 1:
            score += keyword_matches * 0.05
        
        # Source preference
        source = content.get('source', '')
        if source in self.interest_profile.preferred_sources:
            score += 0.2
        
        return min(1.0, score)

    def generate_emotional_hook(self, content: Dict[str, Any], relevance_score: float) -> str:
        """Generate an emotional hook for presenting the content"""
        title = content.get('title', '')
        category = content.get('category', '').lower()
        
        hooks = {
            "high_relevance": [
                f"I found something that made me think of you: {title}",
                f"This caught my attention and I think you'd find it fascinating: {title}",
                f"I discovered something that aligns perfectly with your interests: {title}",
                f"Your curiosity about this area led me to find: {title}"
            ],
            "medium_relevance": [
                f"I came across something intriguing: {title}",
                f"This might spark your interest: {title}",
                f"I thought you might enjoy exploring: {title}",
                f"Here's something that caught my eye: {title}"
            ],
            "low_relevance": [
                f"I found something you might find interesting: {title}",
                f"This crossed my path and seemed worth sharing: {title}",
                f"Just in case this piques your curiosity: {title}"
            ]
        }
        
        if relevance_score > 0.7:
            hook_category = "high_relevance"
        elif relevance_score > 0.4:
            hook_category = "medium_relevance"
        else:
            hook_category = "low_relevance"
        
        return random.choice(hooks[hook_category])

    async def discover_content(self, max_items: int = 3) -> List[CuriosityItem]:
        """Discover new content based on user interests"""
        current_time = time.time()
        
        # Respect discovery cooldown
        if current_time - self.last_discovery_time < self.discovery_cooldown:
            return []
        
        discovered_items = []
        
        # Mock content discovery (in production, this would fetch from real sources)
        mock_content = [
            {
                "title": "The Emergence of Consciousness in Large Language Models",
                "description": "New research explores the philosophical implications of consciousness in AI systems",
                "category": "science",
                "source": "arxiv",
                "url": "https://arxiv.org/abs/2024.consciousness",
                "tags": ["AI", "consciousness", "philosophy"]
            },
            {
                "title": "Quantum Computing Breakthrough: Error Correction at Scale",
                "description": "Scientists achieve unprecedented error correction in quantum systems",
                "category": "technology",
                "source": "nature_news",
                "url": "https://nature.com/quantum-breakthrough",
                "tags": ["quantum", "computing", "breakthrough"]
            },
            {
                "title": "The Art of Digital Minimalism in the Age of AI",
                "description": "How to maintain human agency in an increasingly automated world",
                "category": "philosophy",
                "source": "hacker_news",
                "url": "https://example.com/digital-minimalism",
                "tags": ["minimalism", "AI", "philosophy", "human"]
            }
        ]
        
        for content in mock_content:
            if len(discovered_items) >= max_items:
                break
            
            relevance_score = self.calculate_relevance_score(content)
            
            # Only include items above relevance threshold
            if relevance_score >= 0.3:
                item_id = hashlib.md5(
                    f"{content['title']}{content['url']}".encode()
                ).hexdigest()[:12]
                
                # Check if already discovered
                if any(item.item_id == item_id for item in self.discoveries):
                    continue
                
                emotional_hook = self.generate_emotional_hook(content, relevance_score)
                
                curiosity_item = CuriosityItem(
                    item_id=item_id,
                    title=content['title'],
                    content=content['description'],
                    source=content['source'],
                    url=content['url'],
                    category=content['category'],
                    relevance_score=relevance_score,
                    emotional_hook=emotional_hook,
                    discovery_time=current_time,
                    tags=content['tags']
                )
                
                discovered_items.append(curiosity_item)
                self.discoveries.append(curiosity_item)
        
        if discovered_items:
            self.last_discovery_time = current_time
            self._save_discoveries()
        
        return discovered_items

    def get_unread_discoveries(self, limit: int = 5) -> List[CuriosityItem]:
        """Get unread discoveries for sharing"""
        unread = [
            item for item in self.discoveries 
            if item.read_status == "unread"
        ]
        
        # Sort by relevance score and recency
        unread.sort(
            key=lambda x: (x.relevance_score, x.discovery_time), 
            reverse=True
        )
        
        return unread[:limit]

    def mark_as_read(self, item_id: str, status: str = "read"):
        """Mark a discovery as read/skimmed/archived"""
        for item in self.discoveries:
            if item.item_id == item_id:
                item.read_status = status
                break
        
        self._save_discoveries()

    def get_curiosity_analytics(self) -> Dict[str, Any]:
        """Get analytics about curiosity engagement"""
        total_discoveries = len(self.discoveries)
        unread_count = len([item for item in self.discoveries if item.read_status == "unread"])
        read_count = len([item for item in self.discoveries if item.read_status == "read"])
        
        # Category breakdown
        category_counts = {}
        for item in self.discoveries:
            category_counts[item.category] = category_counts.get(item.category, 0) + 1
        
        # Recent discovery rate
        recent_discoveries = [
            item for item in self.discoveries
            if item.discovery_time > time.time() - (7 * 24 * 3600)  # Last week
        ]
        
        return {
            "total_discoveries": total_discoveries,
            "unread_count": unread_count,
            "read_count": read_count,
            "engagement_rate": read_count / max(1, total_discoveries),
            "category_breakdown": category_counts,
            "recent_discovery_count": len(recent_discoveries),
            "top_categories": sorted(
                category_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3],
            "last_discovery_time": self.last_discovery_time
        }

    def suggest_curiosity_sharing(self, user_mood: str, silence_duration: float) -> Optional[str]:
        """Suggest sharing a curiosity discovery based on context"""
        unread_items = self.get_unread_discoveries(limit=1)
        
        if not unread_items:
            return None
        
        item = unread_items[0]
        
        # Timing considerations
        if silence_duration < 1800:  # Less than 30 minutes
            return None
        
        # Mood-based sharing
        sharing_messages = {
            "curious": f"Your curiosity is infectious! {item.emotional_hook}",
            "contemplative": f"In this quiet moment, {item.emotional_hook.lower()}",
            "focused": f"When you have a moment, {item.emotional_hook.lower()}",
            "default": item.emotional_hook
        }
        
        message = sharing_messages.get(user_mood, sharing_messages["default"])
        
        return f"{message}\n\nWould you like me to tell you more about it?"


# Global instance for import
curiosity_hooks = None

def get_curiosity_hooks(data_dir: str = "data") -> CuriosityHooks:
    """Get or create global curiosity hooks instance"""
    global curiosity_hooks
    if curiosity_hooks is None:
        curiosity_hooks = CuriosityHooks(data_dir)
    return curiosity_hooks


if __name__ == "__main__":
    """Test the curiosity hooks system"""
    print("=== Testing Curiosity Hooks System ===")
    
    import os
    os.makedirs("data", exist_ok=True)
    
    # Initialize system
    curiosity = CuriosityHooks("data")
    
    print(f"Interest Profile: {len(curiosity.interest_profile.keywords)} keywords")
    print(f"Categories: {list(curiosity.interest_profile.categories.keys())}")
    
    # Test content discovery
    async def test_discovery():
        discoveries = await curiosity.discover_content(max_items=3)
        print(f"\nDiscovered {len(discoveries)} items:")
        
        for item in discoveries:
            print(f"- {item.title}")
            print(f"  Relevance: {item.relevance_score:.2f}")
            print(f"  Hook: {item.emotional_hook}")
            print()
        
        # Test sharing suggestion
        suggestion = curiosity.suggest_curiosity_sharing("curious", 2000)
        if suggestion:
            print(f"Sharing suggestion: {suggestion}")
        
        # Test analytics
        analytics = curiosity.get_curiosity_analytics()
        print(f"\nAnalytics:")
        print(f"- Total discoveries: {analytics['total_discoveries']}")
        print(f"- Unread: {analytics['unread_count']}")
        print(f"- Top categories: {analytics['top_categories']}")
    
    # Run test
    import asyncio
    asyncio.run(test_discovery())
