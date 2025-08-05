"""
🌤️ Offline/Online Mode Awareness System
Monitors cloud connectivity and adjusts AI routing accordingly
for the Dolphin AI Orchestrator v2.0
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from handler_registry import handler_registry, HandlerState

class ConnectivityMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class ServiceStatus:
    """Status of an external service"""
    name: str
    url: str
    is_available: bool
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0

class ConnectivityManager:
    """
    Manages online/offline mode detection and routing adjustments
    for cloud-based AI services and external integrations
    """
    
    def __init__(self, analytics_logger, *, on_status_change: Optional[Callable[[Dict[str, Any]], None]] = None, mirror_mode_manager: Any = None):
        self.analytics_logger = analytics_logger
        self.on_status_change = on_status_change
        self.mirror_mode_manager = mirror_mode_manager

        self.log_file = Path("logs/connectivity.log")
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Service endpoints to monitor
        self.services = {
            'openrouter': ServiceStatus(
                name='OpenRouter',
                url='https://openrouter.ai/api/v1/models',
                is_available=True,
                last_check=datetime.now(),
                response_time=0.0
            ),
            'n8n': ServiceStatus(
                name='n8n Workflows',
                url='http://localhost:5678/webhook/health',
                is_available=True,
                last_check=datetime.now(),
                response_time=0.0
            ),
            'kimi': ServiceStatus(
                name='Kimi K2',
                url='https://api.moonshot.cn/v1/models',
                is_available=True,
                last_check=datetime.now(),
                response_time=0.0
            )
        }
        
        # Configuration
        self.check_interval = 30  # seconds
        self.timeout = 10  # 10 seconds
        self.failure_threshold = 3  # Consider offline after 3 consecutive failures
        
        # State
        self.current_mode = ConnectivityMode.UNKNOWN
        self.forced_offline = False
        self.last_mode_change = datetime.now()
        self.is_monitoring = False
        self.emergency_mode = False

        # History
        self.connectivity_history = []
        self.uptime_stats = {}

    def _log(self, message: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
        
    async def start_monitoring(self):
        """Start continuous connectivity monitoring"""
        self.is_monitoring = True
        print("🌤️ Connectivity Manager: Starting monitoring...")
        self._log("Monitoring started")
        
        # Initial check
        await self._check_all_services()
        
        # Start background monitoring
        while self.is_monitoring:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"❌ Connectivity monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop connectivity monitoring"""
        self.is_monitoring = False
        print("⏹️ Connectivity Manager: Stopped monitoring")
        self._log("Monitoring stopped")
    
    async def _check_all_services(self):
        """Check all registered services concurrently"""
        check_tasks = [
            self._check_service(service_id, service)
            for service_id, service in self.services.items()
        ]
        
        await asyncio.gather(*check_tasks, return_exceptions=True)
        await self._update_connectivity_mode()
    
    async def _check_service(self, service_id: str, service: ServiceStatus):
        """Check individual service availability"""
        prev_available = service.is_available
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(service.url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status in [200, 401, 403]:  # Consider auth errors as "available"
                        service.is_available = True
                        service.consecutive_failures = 0
                        service.error_message = None
                    else:
                        service.is_available = False
                        service.consecutive_failures += 1
                        service.error_message = f"HTTP {response.status}"
                    
                    service.response_time = response_time
                    service.last_check = datetime.now()
                    
        except asyncio.TimeoutError:
            service.is_available = False
            service.consecutive_failures += 1
            service.error_message = "Timeout"
            service.response_time = self.timeout
            service.last_check = datetime.now()
            
        except Exception as e:
            service.is_available = False
            service.consecutive_failures += 1
            service.error_message = str(e)
            service.response_time = self.timeout
            service.last_check = datetime.now()
        
        # Update uptime statistics
        self._update_uptime_stats(service_id, service)
        
        # Log the check
        self.analytics_logger.log_custom_event(
            "service_health_check",
            {
                'service': service_id,
                'available': service.is_available,
                'response_time': service.response_time,
                'consecutive_failures': service.consecutive_failures
            }
        )

        previous_available = prev_available

        if (not service.is_available and service.consecutive_failures >= self.failure_threshold and previous_available):
            await self._on_service_down(service_id)
        elif service.is_available and not previous_available:
            await self._on_service_restored(service_id)
        
        print(f"🔍 {service.name}: {'✅' if service.is_available else '❌'} "
              f"({service.response_time:.2f}s)")
    
    def _update_uptime_stats(self, service_id: str, service: ServiceStatus):
        """Update uptime statistics for a service"""
        if service_id not in self.uptime_stats:
            self.uptime_stats[service_id] = {
                'total_checks': 0,
                'successful_checks': 0,
                'last_24h': []
            }
        
        stats = self.uptime_stats[service_id]
        stats['total_checks'] += 1
        
        if service.is_available:
            stats['successful_checks'] += 1
        
        # Track last 24 hours
        current_time = datetime.now()
        stats['last_24h'].append({
            'timestamp': current_time.isoformat(),
            'available': service.is_available
        })
        
        # Clean old entries (keep last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        stats['last_24h'] = [
            entry for entry in stats['last_24h']
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        # Calculate uptime percentage for last 24h
        if stats['last_24h']:
            available_count = sum(1 for entry in stats['last_24h'] if entry['available'])
            service.uptime_percentage = (available_count / len(stats['last_24h'])) * 100

    async def _on_service_down(self, service_id: str) -> None:
        """Handle service becoming unavailable."""
        self._log(f"Service {service_id} marked offline")
        handler_registry.update(service_id, HandlerState.OFFLINE)
        if self.mirror_mode_manager:
            try:
                self.mirror_mode_manager.add_mirror_reflection(
                    "Service outage detected",
                    {"service": service_id}
                )
            except Exception:
                pass
        if self.on_status_change:
            self.on_status_change(self.get_status_summary())

    async def _on_service_restored(self, service_id: str) -> None:
        """Handle service restoration."""
        self._log(f"Service {service_id} back online")
        handler_registry.update(service_id, HandlerState.ONLINE)
        if self.on_status_change:
            self.on_status_change(self.get_status_summary())
    
    async def _update_connectivity_mode(self):
        """Update overall connectivity mode based on service status"""
        if self.forced_offline:
            new_mode = ConnectivityMode.OFFLINE
        else:
            # Check critical services
            openrouter_down = (
                not self.services['openrouter'].is_available or
                self.services['openrouter'].consecutive_failures >= self.failure_threshold
            )
            
            n8n_down = (
                not self.services['n8n'].is_available or
                self.services['n8n'].consecutive_failures >= self.failure_threshold
            )
            
            kimi_down = (
                not self.services['kimi'].is_available or
                self.services['kimi'].consecutive_failures >= self.failure_threshold
            )
            
            # Determine mode
            if openrouter_down and n8n_down:
                new_mode = ConnectivityMode.OFFLINE
            elif openrouter_down or n8n_down:
                new_mode = ConnectivityMode.DEGRADED
            else:
                new_mode = ConnectivityMode.ONLINE
        
        # Check for mode change
        if new_mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = new_mode
            self.last_mode_change = datetime.now()

            self.emergency_mode = new_mode == ConnectivityMode.OFFLINE
            
            # Log mode change
            self.analytics_logger.log_custom_event(
                "connectivity_mode_change",
                {
                    'old_mode': old_mode.value if old_mode else 'unknown',
                    'new_mode': new_mode.value,
                    'reason': self._get_mode_change_reason()
                }
            )
            
            self._log(f"Connectivity mode changed: {old_mode.value if old_mode else 'unknown'} -> {new_mode.value}")
            print(f"🌤️ Connectivity mode changed: {old_mode.value if old_mode else 'unknown'} → {new_mode.value}")

            if self.on_status_change:
                self.on_status_change(self.get_status_summary())
            
            # Add to history
            self.connectivity_history.append({
                'timestamp': datetime.now().isoformat(),
                'mode': new_mode.value,
                'services_status': {
                    service_id: service.is_available
                    for service_id, service in self.services.items()
                }
            })
            
            # Keep last 100 entries
            self.connectivity_history = self.connectivity_history[-100:]
    
    def _get_mode_change_reason(self) -> str:
        """Get human-readable reason for mode change"""
        if self.forced_offline:
            return "Manually forced offline"
        
        down_services = [
            service.name for service in self.services.values()
            if not service.is_available or service.consecutive_failures >= self.failure_threshold
        ]
        
        if down_services:
            return f"Services unavailable: {', '.join(down_services)}"
        else:
            return "All services restored"
    
    def force_offline_mode(self, enabled: bool = True):
        """Manually force offline mode"""
        self.forced_offline = enabled
        print(f"🔒 {'Forced offline mode' if enabled else 'Released offline mode'}")
        
        # Trigger immediate mode update
        asyncio.create_task(self._update_connectivity_mode())
    
    def get_routing_adjustments(self) -> Dict[str, Any]:
        """Get routing adjustments based on current connectivity"""
        adjustments = {
            'mode': self.current_mode.value,
            'disable_openrouter': not self.services['openrouter'].is_available,
            'disable_n8n': not self.services['n8n'].is_available,
            'use_kimi_fallback': True,  # Always available as local fallback
            'prefer_local': self.current_mode in [ConnectivityMode.OFFLINE, ConnectivityMode.DEGRADED],
            'emergency_persona': self.emergency_mode,
            'service_status': {
                service_id: {
                    'available': service.is_available,
                    'response_time': service.response_time,
                    'uptime': service.uptime_percentage
                }
                for service_id, service in self.services.items()
            }
        }
        
        return adjustments
    
    def get_ui_notification(self) -> Optional[Dict[str, Any]]:
        """Get UI notification about connectivity status"""
        if self.current_mode == ConnectivityMode.OFFLINE:
            return {
                'type': 'warning',
                'title': 'Offline Mode Active',
                'message': 'Cloud AI services unavailable. Using local models only.',
                'icon': '🔌',
                'show_in_chat': True
            }
        elif self.current_mode == ConnectivityMode.DEGRADED:
            down_services = [
                service.name for service in self.services.values()
                if not service.is_available
            ]
            return {
                'type': 'info',
                'title': 'Limited Connectivity',
                'message': f'Some services unavailable: {", ".join(down_services)}',
                'icon': '⚠️',
                'show_in_chat': False
            }
        elif self.forced_offline:
            return {
                'type': 'info',
                'title': 'Manual Offline Mode',
                'message': 'You have manually enabled offline mode.',
                'icon': '🔒',
                'show_in_chat': False
            }
        
        return None
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        return {
            'current_mode': self.current_mode.value,
            'forced_offline': self.forced_offline,
            'emergency_mode': self.emergency_mode,
            'last_mode_change': self.last_mode_change.isoformat(),
            'services': {
                service_id: {
                    'name': service.name,
                    'available': service.is_available,
                    'last_check': service.last_check.isoformat(),
                    'response_time': service.response_time,
                    'consecutive_failures': service.consecutive_failures,
                    'uptime_percentage': service.uptime_percentage,
                    'error_message': service.error_message
                }
                for service_id, service in self.services.items()
            },
            'uptime_stats': self.uptime_stats,
            'connectivity_history': self.connectivity_history[-10:],  # Last 10 changes
            'monitoring_active': self.is_monitoring
        }
    
    async def manual_service_check(self, service_id: Optional[str] = None) -> Dict[str, Any]:
        """Manually trigger service check"""
        if service_id and service_id in self.services:
            await self._check_service(service_id, self.services[service_id])
            return {service_id: self.services[service_id]}
        else:
            await self._check_all_services()
            return {sid: service for sid, service in self.services.items()}

# Global connectivity manager instance
connectivity_manager = None

def get_connectivity_manager():
    """Get the global connectivity manager instance"""
    return connectivity_manager

def initialize_connectivity_manager(analytics_logger, **kwargs):
    """Initialize the global connectivity manager"""
    global connectivity_manager
    connectivity_manager = ConnectivityManager(analytics_logger, **kwargs)
    return connectivity_manager
