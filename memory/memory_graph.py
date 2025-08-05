"""
True Recall - Memory Graph

This module manages the storage and linking of episodic events in a graph structure.
It handles the relationships between events and provides efficient retrieval mechanisms.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import re

logger = logging.getLogger(__name__)

class MemoryGraph:
    """
    Manages episodic memory events and their relationships in a graph structure.
    
    Provides efficient storage, linking, and retrieval of memory events with
    temporal, semantic, and causal relationships.
    """
    
    def __init__(self, memory_store):
        """Initialize the memory graph."""
        self.memory_store = memory_store
        self.event_cache: Dict[str, Any] = {}  # LRU cache for recent events
        self.relationship_cache: Dict[str, List[str]] = {}  # Cache for relationships
        self.max_cache_size = 1000
        
        logger.info("🕸️ Memory Graph initialized")
    
    async def add_event(self, memory_event) -> bool:
        """
        Add a new memory event to the graph.
        
        Args:
            memory_event: MemoryEvent object to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Convert to dict for storage
            event_dict = asdict(memory_event)
            
            # Store the event
            success = await self.memory_store.store_event(event_dict)
            
            if success:
                # Update cache
                self._add_to_cache(memory_event.id, event_dict)
                
                # Update relationship mappings
                await self._update_relationships(memory_event)
                
                logger.info(f"📝 Added event to graph: {memory_event.id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to add event to graph: {e}")
            return False
    
    async def get_event(self, event_id: str):
        """Get a specific event by ID."""
        try:
            # Check cache first
            if event_id in self.event_cache:
                event_dict = self.event_cache[event_id]
                return self._dict_to_memory_event(event_dict)
            
            # Fetch from storage
            event_dict = await self.memory_store.get_event(event_id)
            if event_dict:
                self._add_to_cache(event_id, event_dict)
                return self._dict_to_memory_event(event_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get event {event_id}: {e}")
            return None
    
    async def search_events(self,
                           query: Optional[str] = None,
                           actor: Optional[str] = None,
                           event_type: Optional[str] = None,
                           emotion_tags: Optional[List[str]] = None,
                           time_range: Optional[Tuple[datetime, datetime]] = None,
                           min_salience: float = 0.0,
                           limit: int = 20) -> List[Any]:
        """
        Search for events based on various criteria.
        
        Args:
            query: Text search query
            actor: Filter by actor
            event_type: Filter by event type
            emotion_tags: Filter by emotion tags
            time_range: Time range tuple (start, end)
            min_salience: Minimum salience threshold
            limit: Maximum results to return
            
        Returns:
            List of MemoryEvent objects matching criteria
        """
        try:
            # Build search criteria
            criteria = {
                'actor': actor,
                'event_type': event_type,
                'emotion_tags': emotion_tags,
                'min_salience': min_salience,
                'limit': limit
            }
            
            if time_range:
                criteria['start_time'] = time_range[0].isoformat()
                criteria['end_time'] = time_range[1].isoformat()
            
            # Perform search in storage
            event_dicts = await self.memory_store.retrieve_events(
                time_range=time_range,
                actor=actor,
                event_type=event_type,
                min_salience=min_salience,
                emotion_tags=emotion_tags,
                limit=limit
            )
            
            # Convert to MemoryEvent objects
            events = []
            for event_dict in event_dicts:
                memory_event = self._dict_to_memory_event(event_dict)
                if memory_event:
                    events.append(memory_event)
                    # Cache the event
                    self._add_to_cache(memory_event.id, event_dict)
            
            # Sort by salience and recency
            events.sort(key=lambda e: (e.salience, e.timestamp), reverse=True)
            
            logger.info(f"🔍 Found {len(events)} events matching search criteria")
            return events[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to search events: {e}")
            return []
    
    async def find_related_events(self,
                                 content: str,
                                 context_ids: List[str],
                                 limit: int = 5) -> List[Any]:
        """
        Find events related to given content and context.
        
        Args:
            content: Content to find relations for
            context_ids: List of context event IDs
            limit: Maximum number of related events
            
        Returns:
            List of related MemoryEvent objects
        """
        try:
            # Extract keywords from content
            keywords = self._extract_keywords(content)
            
            # Search for events with similar keywords
            related_events = []
            
            # Search by keywords
            if keywords:
                keyword_query = " ".join(keywords[:5])  # Top 5 keywords
                keyword_events = await self.search_events(
                    query=keyword_query,
                    limit=limit * 2
                )
                related_events.extend(keyword_events)
            
            # Get events from context IDs
            for context_id in context_ids:
                context_event = await self.get_event(context_id)
                if context_event:
                    related_events.append(context_event)
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_events = []
            for event in related_events:
                if event.id not in seen_ids:
                    seen_ids.add(event.id)
                    unique_events.append(event)
                    if len(unique_events) >= limit:
                        break
            
            logger.info(f"🔗 Found {len(unique_events)} related events")
            return unique_events
            
        except Exception as e:
            logger.error(f"❌ Failed to find related events: {e}")
            return []
    
    async def get_temporal_context(self,
                                  reference_event,
                                  start_time: datetime,
                                  end_time: datetime,
                                  context_window: int = 5) -> List[Any]:
        """
        Get temporal context events around a reference event.
        
        Args:
            reference_event: The reference MemoryEvent
            start_time: Start of time window
            end_time: End of time window
            context_window: Number of events to include
            
        Returns:
            List of contextual MemoryEvent objects
        """
        try:
            # Get events in time range
            context_events = await self.search_events(
                time_range=(start_time, end_time),
                limit=context_window * 3  # Get more to filter
            )
            
            # Remove the reference event itself
            context_events = [e for e in context_events if e.id != reference_event.id]
            
            # Sort by temporal proximity to reference event
            ref_time = datetime.fromisoformat(reference_event.timestamp)
            context_events.sort(
                key=lambda e: abs((datetime.fromisoformat(e.timestamp) - ref_time).total_seconds())
            )
            
            return context_events[:context_window]
            
        except Exception as e:
            logger.error(f"❌ Failed to get temporal context: {e}")
            return []
    
    async def get_related_events(self, event_id: str, depth: int = 1) -> List[Any]:
        """
        Get events related to a specific event ID.
        
        Args:
            event_id: The event ID to find relations for
            depth: Depth of relationship traversal
            
        Returns:
            List of related MemoryEvent objects
        """
        try:
            if event_id in self.relationship_cache:
                related_ids = self.relationship_cache[event_id]
            else:
                related_ids = await self._build_relationship_map(event_id)
            
            related_events = []
            for related_id in related_ids:
                event = await self.get_event(related_id)
                if event:
                    related_events.append(event)
            
            # If depth > 1, recursively get related events
            if depth > 1:
                for event in related_events[:]:  # Copy to avoid modification during iteration
                    deeper_related = await self.get_related_events(event.id, depth - 1)
                    for deep_event in deeper_related:
                        if deep_event.id not in [e.id for e in related_events]:
                            related_events.append(deep_event)
            
            return related_events
            
        except Exception as e:
            logger.error(f"❌ Failed to get related events: {e}")
            return []
    
    async def get_event_chain(self, start_event_id: str, max_length: int = 10) -> List[Any]:
        """
        Get a chain of events starting from a specific event.
        
        Args:
            start_event_id: Starting event ID
            max_length: Maximum chain length
            
        Returns:
            List of events in chronological order
        """
        try:
            chain = []
            current_event = await self.get_event(start_event_id)
            
            if not current_event:
                return chain
            
            chain.append(current_event)
            
            # Follow related events chronologically
            for _ in range(max_length - 1):
                related_events = await self.get_related_events(current_event.id)
                
                # Find the next event chronologically
                current_time = datetime.fromisoformat(current_event.timestamp)
                next_event = None
                
                for event in related_events:
                    event_time = datetime.fromisoformat(event.timestamp)
                    if event_time > current_time:
                        if next_event is None or event_time < datetime.fromisoformat(next_event.timestamp):
                            next_event = event
                
                if next_event and next_event.id not in [e.id for e in chain]:
                    chain.append(next_event)
                    current_event = next_event
                else:
                    break
            
            logger.info(f"🔗 Built event chain of length {len(chain)}")
            return chain
            
        except Exception as e:
            logger.error(f"❌ Failed to build event chain: {e}")
            return []
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory graph."""
        try:
            total_events = await self.memory_store.get_event_count()
            
            # Get recent events for analysis
            recent_time = datetime.now() - timedelta(days=7)
            recent_events = await self.search_events(
                time_range=(recent_time, datetime.now()),
                limit=1000
            )
            
            # Analyze relationship density
            total_relationships = sum(len(event.related_ids) for event in recent_events)
            avg_relationships = total_relationships / len(recent_events) if recent_events else 0
            
            return {
                'total_events': total_events,
                'cache_size': len(self.event_cache),
                'recent_events': len(recent_events),
                'average_relationships': round(avg_relationships, 2),
                'relationship_cache_size': len(self.relationship_cache)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get graph stats: {e}")
            return {}
    
    def _dict_to_memory_event(self, event_dict: Dict[str, Any]):
        """Convert dictionary to MemoryEvent object."""
        try:
            # Import here to avoid circular imports
            from .recall_engine import MemoryEvent
            
            return MemoryEvent(
                id=event_dict['id'],
                timestamp=event_dict['timestamp'],
                actor=event_dict['actor'],
                event_type=event_dict['event_type'],
                content=event_dict['content'],
                tone=event_dict['tone'],
                emotion_tags=event_dict['emotion_tags'],
                salience=event_dict['salience'],
                related_ids=event_dict['related_ids'],
                metadata=event_dict.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"❌ Failed to convert dict to MemoryEvent: {e}")
            return None
    
    def _add_to_cache(self, event_id: str, event_dict: Dict[str, Any]):
        """Add event to LRU cache."""
        # Remove oldest if cache is full
        if len(self.event_cache) >= self.max_cache_size:
            # Remove oldest entry (this is a simple implementation)
            oldest_key = next(iter(self.event_cache))
            del self.event_cache[oldest_key]
        
        self.event_cache[event_id] = event_dict
    
    async def _update_relationships(self, memory_event):
        """Update relationship mappings for an event."""
        try:
            # Build bidirectional relationships
            for related_id in memory_event.related_ids:
                if related_id not in self.relationship_cache:
                    self.relationship_cache[related_id] = []
                
                if memory_event.id not in self.relationship_cache[related_id]:
                    self.relationship_cache[related_id].append(memory_event.id)
            
            # Update current event's relationships
            self.relationship_cache[memory_event.id] = memory_event.related_ids.copy()
            
        except Exception as e:
            logger.error(f"❌ Failed to update relationships: {e}")
    
    async def _build_relationship_map(self, event_id: str) -> List[str]:
        """Build relationship map for an event."""
        try:
            event = await self.get_event(event_id)
            if not event:
                return []
            
            related_ids = event.related_ids.copy()
            
            # Find additional relationships through content similarity
            content_related = await self._find_content_related(event)
            for related_event in content_related:
                if related_event.id not in related_ids:
                    related_ids.append(related_event.id)
            
            # Cache the result
            self.relationship_cache[event_id] = related_ids
            return related_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to build relationship map: {e}")
            return []
    
    async def _find_content_related(self, event, max_results: int = 5) -> List[Any]:
        """Find events related by content similarity."""
        try:
            keywords = self._extract_keywords(event.content)
            if not keywords:
                return []
            
            # Search for events with similar keywords
            query = " ".join(keywords[:3])  # Use top 3 keywords
            similar_events = await self.search_events(
                query=query,
                actor=event.actor,  # Prefer same actor
                limit=max_results * 2
            )
            
            # Filter out the current event and low-salience events
            related_events = [
                e for e in similar_events 
                if e.id != event.id and e.salience > 0.3
            ]
            
            return related_events[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Failed to find content-related events: {e}")
            return []
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content for relationship finding."""
        try:
            # Simple keyword extraction
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            }
            
            # Extract words, remove punctuation, filter stop words
            words = re.findall(r'\b\w+\b', content.lower())
            keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Return unique keywords, prioritizing by length (longer words often more meaningful)
            unique_keywords = list(dict.fromkeys(keywords))  # Preserve order while removing duplicates
            unique_keywords.sort(key=len, reverse=True)
            
            return unique_keywords[:10]  # Top 10 keywords
            
        except Exception as e:
            logger.error(f"❌ Failed to extract keywords: {e}")
            return []
    
    async def export_graph_structure(self) -> Dict[str, Any]:
        """Export the graph structure for analysis or backup."""
        try:
            # Get all events
            all_events = await self.memory_store.get_all_events()
            
            # Build adjacency list representation
            adjacency_list = {}
            node_data = {}
            
            for event_dict in all_events:
                event_id = event_dict['id']
                adjacency_list[event_id] = event_dict.get('related_ids', [])
                node_data[event_id] = {
                    'timestamp': event_dict['timestamp'],
                    'actor': event_dict['actor'],
                    'event_type': event_dict['event_type'],
                    'salience': event_dict['salience'],
                    'emotion_tags': event_dict['emotion_tags']
                }
            
            return {
                'graph_type': 'memory_graph',
                'export_timestamp': datetime.now().isoformat(),
                'node_count': len(node_data),
                'edge_count': sum(len(edges) for edges in adjacency_list.values()),
                'adjacency_list': adjacency_list,
                'node_data': node_data
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export graph structure: {e}")
            return {}
    
    async def analyze_graph_properties(self) -> Dict[str, Any]:
        """Analyze mathematical properties of the memory graph."""
        try:
            graph_data = await self.export_graph_structure()
            adjacency_list = graph_data.get('adjacency_list', {})
            
            if not adjacency_list:
                return {}
            
            # Calculate basic graph properties
            node_count = len(adjacency_list)
            edge_count = sum(len(edges) for edges in adjacency_list.values())
            
            # Calculate degree distribution
            degrees = [len(edges) for edges in adjacency_list.values()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0
            
            # Find connected components (simplified)
            visited = set()
            components = []
            
            def dfs(node, component):
                if node in visited:
                    return
                visited.add(node)
                component.append(node)
                for neighbor in adjacency_list.get(node, []):
                    if neighbor in adjacency_list:  # Ensure neighbor exists
                        dfs(neighbor, component)
            
            for node in adjacency_list:
                if node not in visited:
                    component = []
                    dfs(node, component)
                    components.append(component)
            
            # Calculate clustering coefficient (simplified local clustering)
            clustering_coeffs = []
            for node, neighbors in adjacency_list.items():
                if len(neighbors) < 2:
                    clustering_coeffs.append(0)
                    continue
                
                # Count triangles
                triangles = 0
                for i, neighbor1 in enumerate(neighbors):
                    for j, neighbor2 in enumerate(neighbors[i+1:], i+1):
                        if neighbor2 in adjacency_list.get(neighbor1, []):
                            triangles += 1
                
                possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
                clustering = triangles / possible_triangles if possible_triangles > 0 else 0
                clustering_coeffs.append(clustering)
            
            avg_clustering = sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0
            
            return {
                'node_count': node_count,
                'edge_count': edge_count,
                'average_degree': round(avg_degree, 2),
                'max_degree': max_degree,
                'connected_components': len(components),
                'largest_component_size': max(len(comp) for comp in components) if components else 0,
                'average_clustering_coefficient': round(avg_clustering, 3),
                'density': round(edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0, 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze graph properties: {e}")
            return {}
