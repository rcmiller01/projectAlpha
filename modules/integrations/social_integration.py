"""
Social Media Integration for AI Companion

This module provides read-only social media monitoring for:
- Reddit (user interests, trending topics)
- X/Twitter (timeline, mentions)
- Facebook (timeline, notifications)
- Threads (timeline, conversations)
- Instagram (feed, stories - read-only)

Focuses on understanding user interests and social context
while maintaining privacy and read-only access.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import re
from enum import Enum

logger = logging.getLogger(__name__)

class PlatformType(Enum):
    REDDIT = "reddit"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    THREADS = "threads"
    INSTAGRAM = "instagram"

@dataclass
class SocialPost:
    """Individual social media post"""
    platform: PlatformType
    post_id: str
    content: str
    author: str
    timestamp: datetime
    engagement: Dict[str, int]  # likes, shares, comments, etc.
    topics: List[str]
    sentiment: Optional[str] = None  # positive, negative, neutral
    relevance_score: float = 0.0  # 0.0 to 1.0

@dataclass
class UserInterest:
    """Detected user interest from social activity"""
    topic: str
    platforms: Set[PlatformType]
    frequency: int
    last_seen: datetime
    confidence: float  # 0.0 to 1.0
    related_keywords: List[str]

@dataclass
class SocialContext:
    """Current social media context"""
    active_topics: List[str]
    trending_interests: List[UserInterest]
    recent_interactions: List[SocialPost]
    mood_indicators: Dict[str, float]  # emotional indicators from social activity
    last_updated: datetime

class SocialMediaConnector:
    """Base connector for social media platforms"""
    
    def __init__(self, config_path: str = "data/social_config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.authenticated_platforms = set()
        self.rate_limits = {}
        self.last_fetch = {}
        self.load_config()
        
    def load_config(self):
        """Load social media configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.authenticated_platforms = set(self.config.get('authenticated_platforms', []))
                logger.info(f"Loaded social config for {len(self.authenticated_platforms)} platforms")
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading social config: {e}")
            self.config = self._get_default_config()
    
    def save_config(self):
        """Save social media configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving social config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default social media configuration"""
        return {
            "authenticated_platforms": [],
            "privacy_mode": "read_only",
            "content_filters": {
                "min_relevance_score": 0.3,
                "blocked_keywords": ["nsfw", "spam"],
                "preferred_topics": []
            },
            "rate_limits": {
                "reddit": {"requests_per_hour": 60, "batch_size": 25},
                "twitter": {"requests_per_hour": 180, "batch_size": 20},
                "facebook": {"requests_per_hour": 120, "batch_size": 15},
                "threads": {"requests_per_hour": 100, "batch_size": 15},
                "instagram": {"requests_per_hour": 60, "batch_size": 10}
            },
            "fetch_intervals": {
                "reddit": 900,  # 15 minutes
                "twitter": 300,  # 5 minutes
                "facebook": 1800,  # 30 minutes
                "threads": 600,  # 10 minutes
                "instagram": 3600  # 1 hour
            },
            "interest_tracking": True,
            "sentiment_analysis": True,
            "trend_detection": True
        }

class RedditConnector:
    """Reddit read-only connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.subreddits = set()
        self.user_interests = []
        
    async def authenticate(self) -> bool:
        """Authenticate with Reddit API (read-only)"""
        try:
            # In real implementation, use Reddit API credentials
            logger.info("Authenticating with Reddit API")
            return True
        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
            return False
    
    async def fetch_user_timeline(self, limit: int = 25) -> List[SocialPost]:
        """Fetch user's Reddit timeline"""
        posts = []
        try:
            # Simulate Reddit API call
            # In real implementation, use PRAW (Python Reddit API Wrapper)
            
            # Simulated Reddit posts
            sample_posts = [
                {
                    "id": "reddit_001",
                    "title": "Interesting discussion about AI ethics",
                    "content": "Found this fascinating thread about AI consciousness...",
                    "subreddit": "r/artificial",
                    "author": "ai_enthusiast",
                    "score": 156,
                    "num_comments": 23,
                    "created_utc": datetime.now() - timedelta(hours=2)
                },
                {
                    "id": "reddit_002", 
                    "title": "Beautiful sunset photo",
                    "content": "Captured this amazing sunset yesterday...",
                    "subreddit": "r/EarthPorn",
                    "author": "photographer",
                    "score": 89,
                    "num_comments": 12,
                    "created_utc": datetime.now() - timedelta(hours=4)
                }
            ]
            
            for post_data in sample_posts:
                post = SocialPost(
                    platform=PlatformType.REDDIT,
                    post_id=post_data["id"],
                    content=f"{post_data['title']}: {post_data['content']}",
                    author=post_data["author"],
                    timestamp=post_data["created_utc"],
                    engagement={
                        "upvotes": post_data["score"],
                        "comments": post_data["num_comments"]
                    },
                    topics=self._extract_topics(post_data["title"] + " " + post_data["content"]),
                    relevance_score=0.8
                )
                posts.append(post)
            
            logger.debug(f"Fetched {len(posts)} Reddit posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit timeline: {e}")
            return []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from Reddit content"""
        topics = []
        # Simple keyword extraction (in real implementation, use NLP)
        keywords = ["AI", "artificial intelligence", "photography", "sunset", "technology", "ethics"]
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                topics.append(keyword)
        
        return topics

class TwitterConnector:
    """X/Twitter read-only connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def authenticate(self) -> bool:
        """Authenticate with Twitter API v2"""
        try:
            # In real implementation, use Twitter API v2 credentials
            logger.info("Authenticating with Twitter API")
            return True
        except Exception as e:
            logger.error(f"Twitter authentication failed: {e}")
            return False
    
    async def fetch_user_timeline(self, limit: int = 20) -> List[SocialPost]:
        """Fetch user's Twitter timeline"""
        posts = []
        try:
            # Simulate Twitter API call
            # In real implementation, use tweepy or twitter-api-v2
            
            sample_tweets = [
                {
                    "id": "twitter_001",
                    "text": "Just saw an amazing demo of the new AI assistant. The emotional intelligence is incredible! #AI #Technology",
                    "author": "tech_reviewer",
                    "created_at": datetime.now() - timedelta(minutes=30),
                    "public_metrics": {"like_count": 45, "retweet_count": 12, "reply_count": 8}
                },
                {
                    "id": "twitter_002",
                    "text": "Morning coffee ritual complete ☕ Ready to tackle the day!",
                    "author": "daily_user",
                    "created_at": datetime.now() - timedelta(hours=1),
                    "public_metrics": {"like_count": 23, "retweet_count": 2, "reply_count": 4}
                }
            ]
            
            for tweet_data in sample_tweets:
                post = SocialPost(
                    platform=PlatformType.TWITTER,
                    post_id=tweet_data["id"],
                    content=tweet_data["text"],
                    author=tweet_data["author"],
                    timestamp=tweet_data["created_at"],
                    engagement={
                        "likes": tweet_data["public_metrics"]["like_count"],
                        "retweets": tweet_data["public_metrics"]["retweet_count"],
                        "replies": tweet_data["public_metrics"]["reply_count"]
                    },
                    topics=self._extract_hashtags(tweet_data["text"]),
                    sentiment=self._analyze_sentiment(tweet_data["text"]),
                    relevance_score=0.7
                )
                posts.append(post)
            
            logger.debug(f"Fetched {len(posts)} Twitter posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Twitter timeline: {e}")
            return []
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from tweet"""
        hashtags = re.findall(r'#(\w+)', text)
        return hashtags
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ["amazing", "incredible", "love", "great", "awesome", "happy"]
        negative_words = ["hate", "terrible", "awful", "bad", "sad", "angry"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

class FacebookConnector:
    """Facebook read-only connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def authenticate(self) -> bool:
        """Authenticate with Facebook Graph API"""
        try:
            # In real implementation, use Facebook Graph API
            logger.info("Authenticating with Facebook API")
            return True
        except Exception as e:
            logger.error(f"Facebook authentication failed: {e}")
            return False
    
    async def fetch_user_timeline(self, limit: int = 15) -> List[SocialPost]:
        """Fetch user's Facebook timeline"""
        posts = []
        try:
            # Simulate Facebook API call
            sample_posts = [
                {
                    "id": "fb_001",
                    "message": "Had a wonderful dinner with family tonight. Feeling grateful! 🙏",
                    "from": {"name": "family_user"},
                    "created_time": datetime.now() - timedelta(hours=3),
                    "reactions": {"summary": {"total_count": 34}},
                    "comments": {"summary": {"total_count": 7}}
                }
            ]
            
            for post_data in sample_posts:
                post = SocialPost(
                    platform=PlatformType.FACEBOOK,
                    post_id=post_data["id"],
                    content=post_data.get("message", ""),
                    author=post_data["from"]["name"],
                    timestamp=post_data["created_time"],
                    engagement={
                        "reactions": post_data["reactions"]["summary"]["total_count"],
                        "comments": post_data["comments"]["summary"]["total_count"]
                    },
                    topics=["family", "gratitude"],
                    sentiment="positive",
                    relevance_score=0.6
                )
                posts.append(post)
            
            logger.debug(f"Fetched {len(posts)} Facebook posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Facebook timeline: {e}")
            return []

class ThreadsConnector:
    """Threads read-only connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def authenticate(self) -> bool:
        """Authenticate with Threads API"""
        try:
            # Threads uses Instagram Basic Display API for now
            logger.info("Authenticating with Threads API")
            return True
        except Exception as e:
            logger.error(f"Threads authentication failed: {e}")
            return False
    
    async def fetch_user_timeline(self, limit: int = 15) -> List[SocialPost]:
        """Fetch user's Threads timeline"""
        posts = []
        try:
            # Simulate Threads API call
            sample_threads = [
                {
                    "id": "threads_001",
                    "text": "The intersection of AI and creativity is fascinating. Seeing more artists collaborate with AI tools.",
                    "username": "creative_mind",
                    "timestamp": datetime.now() - timedelta(minutes=45),
                    "like_count": 28,
                    "reply_count": 6
                }
            ]
            
            for thread_data in sample_threads:
                post = SocialPost(
                    platform=PlatformType.THREADS,
                    post_id=thread_data["id"],
                    content=thread_data["text"],
                    author=thread_data["username"],
                    timestamp=thread_data["timestamp"],
                    engagement={
                        "likes": thread_data["like_count"],
                        "replies": thread_data["reply_count"]
                    },
                    topics=["AI", "creativity", "art"],
                    sentiment="neutral",
                    relevance_score=0.8
                )
                posts.append(post)
            
            logger.debug(f"Fetched {len(posts)} Threads posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Threads timeline: {e}")
            return []

class InstagramConnector:
    """Instagram read-only connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def authenticate(self) -> bool:
        """Authenticate with Instagram Basic Display API"""
        try:
            # In real implementation, use Instagram Basic Display API
            logger.info("Authenticating with Instagram API")
            return True
        except Exception as e:
            logger.error(f"Instagram authentication failed: {e}")
            return False
    
    async def fetch_user_timeline(self, limit: int = 10) -> List[SocialPost]:
        """Fetch user's Instagram feed"""
        posts = []
        try:
            # Simulate Instagram API call
            sample_posts = [
                {
                    "id": "ig_001",
                    "caption": "Beautiful morning walk in the park 🌅 #nature #morning #peaceful",
                    "username": "nature_lover",
                    "timestamp": datetime.now() - timedelta(hours=6),
                    "like_count": 156,
                    "comments_count": 23
                }
            ]
            
            for post_data in sample_posts:
                post = SocialPost(
                    platform=PlatformType.INSTAGRAM,
                    post_id=post_data["id"],
                    content=post_data.get("caption", ""),
                    author=post_data["username"],
                    timestamp=post_data["timestamp"],
                    engagement={
                        "likes": post_data["like_count"],
                        "comments": post_data["comments_count"]
                    },
                    topics=["nature", "morning", "peaceful"],
                    sentiment="positive",
                    relevance_score=0.5
                )
                posts.append(post)
            
            logger.debug(f"Fetched {len(posts)} Instagram posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Instagram timeline: {e}")
            return []

class SocialInterestAnalyzer:
    """Analyzes social media activity to determine user interests"""
    
    def __init__(self):
        self.interest_cache = {}
        self.topic_frequency = {}
        
    def analyze_posts(self, posts: List[SocialPost]) -> List[UserInterest]:
        """Analyze posts to extract user interests"""
        interests = []
        
        try:
            # Track topic frequency across platforms
            platform_topics = {}
            
            for post in posts:
                for topic in post.topics:
                    if topic not in platform_topics:
                        platform_topics[topic] = {"platforms": set(), "frequency": 0, "last_seen": post.timestamp}
                    
                    platform_topics[topic]["platforms"].add(post.platform)
                    platform_topics[topic]["frequency"] += 1
                    platform_topics[topic]["last_seen"] = max(platform_topics[topic]["last_seen"], post.timestamp)
            
            # Convert to UserInterest objects
            for topic, data in platform_topics.items():
                if data["frequency"] >= 2:  # Minimum frequency threshold
                    confidence = min(1.0, data["frequency"] / 10.0)  # Normalize confidence
                    
                    interest = UserInterest(
                        topic=topic,
                        platforms=data["platforms"],
                        frequency=data["frequency"],
                        last_seen=data["last_seen"],
                        confidence=confidence,
                        related_keywords=self._get_related_keywords(topic)
                    )
                    interests.append(interest)
            
            # Sort by confidence and frequency
            interests.sort(key=lambda x: (x.confidence, x.frequency), reverse=True)
            
            logger.debug(f"Analyzed {len(interests)} user interests")
            return interests
            
        except Exception as e:
            logger.error(f"Error analyzing user interests: {e}")
            return []
    
    def _get_related_keywords(self, topic: str) -> List[str]:
        """Get related keywords for a topic"""
        keyword_map = {
            "AI": ["artificial intelligence", "machine learning", "neural networks", "automation"],
            "technology": ["tech", "innovation", "digital", "software"],
            "photography": ["camera", "lens", "composition", "lighting"],
            "nature": ["outdoors", "wildlife", "landscape", "environment"],
            "creativity": ["art", "design", "inspiration", "artistic"],
            "family": ["relationships", "loved ones", "together", "bonding"]
        }
        return keyword_map.get(topic, [])
    
    def get_mood_indicators(self, posts: List[SocialPost]) -> Dict[str, float]:
        """Extract mood indicators from social activity"""
        mood_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        total_posts = len(posts)
        
        if total_posts == 0:
            return mood_scores
        
        for post in posts:
            if post.sentiment:
                mood_scores[post.sentiment] += 1.0
        
        # Normalize scores
        for mood in mood_scores:
            mood_scores[mood] /= total_posts
        
        return mood_scores

class SocialMediaIntegration:
    """Main social media integration manager"""
    
    def __init__(self):
        self.connector = SocialMediaConnector()
        self.reddit = RedditConnector(self.connector.config)
        self.twitter = TwitterConnector(self.connector.config)
        self.facebook = FacebookConnector(self.connector.config)
        self.threads = ThreadsConnector(self.connector.config)
        self.instagram = InstagramConnector(self.connector.config)
        self.analyzer = SocialInterestAnalyzer()
        
        self.platform_connectors = {
            PlatformType.REDDIT: self.reddit,
            PlatformType.TWITTER: self.twitter,
            PlatformType.FACEBOOK: self.facebook,
            PlatformType.THREADS: self.threads,
            PlatformType.INSTAGRAM: self.instagram
        }
    
    async def initialize(self, platforms: List[str]) -> Dict[str, bool]:
        """Initialize social media connections"""
        results = {}
        
        for platform_name in platforms:
            try:
                platform_type = PlatformType(platform_name.lower())
                connector = self.platform_connectors.get(platform_type)
                
                if connector:
                    success = await connector.authenticate()
                    results[platform_name] = success
                    
                    if success:
                        self.connector.authenticated_platforms.add(platform_name.lower())
                else:
                    results[platform_name] = False
                    
            except Exception as e:
                logger.error(f"Failed to initialize {platform_name}: {e}")
                results[platform_name] = False
        
        # Update config
        self.connector.config['authenticated_platforms'] = list(self.connector.authenticated_platforms)
        self.connector.save_config()
        
        logger.info(f"Initialized {sum(results.values())} of {len(platforms)} social platforms")
        return results
    
    async def get_social_context(self) -> SocialContext:
        """Get comprehensive social media context"""
        try:
            all_posts = []
            
            # Fetch from all authenticated platforms
            for platform_name in self.connector.authenticated_platforms:
                try:
                    platform_type = PlatformType(platform_name)
                    connector = self.platform_connectors.get(platform_type)
                    
                    if connector:
                        posts = await connector.fetch_user_timeline()
                        all_posts.extend(posts)
                        
                except Exception as e:
                    logger.error(f"Error fetching from {platform_name}: {e}")
            
            # Analyze interests
            user_interests = self.analyzer.analyze_posts(all_posts)
            
            # Get mood indicators
            mood_indicators = self.analyzer.get_mood_indicators(all_posts)
            
            # Extract active topics
            active_topics = list(set([topic for post in all_posts for topic in post.topics]))
            
            # Sort recent interactions by timestamp
            recent_interactions = sorted(all_posts, key=lambda x: x.timestamp, reverse=True)[:20]
            
            context = SocialContext(
                active_topics=active_topics,
                trending_interests=user_interests,
                recent_interactions=recent_interactions,
                mood_indicators=mood_indicators,
                last_updated=datetime.now()
            )
            
            logger.info(f"Generated social context with {len(all_posts)} posts from {len(self.connector.authenticated_platforms)} platforms")
            return context
            
        except Exception as e:
            logger.error(f"Error getting social context: {e}")
            return SocialContext(
                active_topics=[],
                trending_interests=[],
                recent_interactions=[],
                mood_indicators={},
                last_updated=datetime.now()
            )
    
    async def get_platform_status(self) -> Dict[str, Any]:
        """Get status of all platform connections"""
        return {
            "authenticated_platforms": list(self.connector.authenticated_platforms),
            "total_platforms": len(self.platform_connectors),
            "privacy_mode": self.connector.config.get("privacy_mode", "read_only"),
            "last_fetch": self.connector.last_fetch,
            "rate_limits": self.connector.config.get("rate_limits", {})
        }

# Example usage and testing
async def demo_social_integration():
    """Demonstrate social media integration"""
    social_integration = SocialMediaIntegration()
    
    print("=== Social Media Integration Demo ===")
    
    # Initialize platforms
    platforms = ["reddit", "twitter", "facebook", "threads", "instagram"]
    init_results = await social_integration.initialize(platforms)
    
    print(f"Platform initialization results:")
    for platform, success in init_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {platform.title()}")
    
    # Get social context
    context = await social_integration.get_social_context()
    
    print(f"\nSocial Context:")
    print(f"  Active Topics: {', '.join(context.active_topics[:5])}")
    print(f"  Trending Interests: {len(context.trending_interests)}")
    
    for interest in context.trending_interests[:3]:
        platforms_str = ", ".join([p.value for p in interest.platforms])
        print(f"    • {interest.topic} (confidence: {interest.confidence:.2f}, platforms: {platforms_str})")
    
    print(f"  Recent Interactions: {len(context.recent_interactions)}")
    print(f"  Mood Indicators: {context.mood_indicators}")
    
    # Status check
    status = await social_integration.get_platform_status()
    print(f"\nStatus: {status}")

if __name__ == "__main__":
    asyncio.run(demo_social_integration())
