"""
N8N Workflow Integration Module

Handles automation workflows with persona oversight and emotional intelligence.
Provides secure, monitored automation for daily activities.
"""

import asyncio
import json
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import os

class WorkflowType(Enum):
    EMAIL_AUTOMATION = "email_automation"
    CALENDAR_SYNC = "calendar_sync"
    TASK_CREATION = "task_creation"
    FILE_ORGANIZATION = "file_organization"
    NOTIFICATION_ROUTING = "notification_routing"
    DATA_BACKUP = "data_backup"
    SOCIAL_MEDIA = "social_media"
    RESEARCH_AUTOMATION = "research_automation"

class WorkflowStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"

class MonitoringLevel(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PERSONA_SUPERVISED = "persona_supervised"

@dataclass
class WorkflowTrigger:
    trigger_type: str
    conditions: Dict[str, Any]
    persona_approval_required: bool
    safety_checks: List[str]

@dataclass
class WorkflowAction:
    action_type: str
    parameters: Dict[str, Any]
    rollback_possible: bool
    risk_level: str  # low, medium, high
    persona_guidance: str

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: WorkflowStatus
    persona_supervisor: str
    results: Dict[str, Any]
    errors: List[str]
    persona_interventions: List[Dict[str, Any]]

class N8NWorkflowManager:
    """
    Manages N8N workflow integrations with persona oversight and safety monitoring.
    Provides emotional intelligence and context awareness to automation.
    """
    
    def __init__(self, n8n_base_url: str, api_key: str, user_id: str):
        self.n8n_base_url = n8n_base_url.rstrip('/')
        self.api_key = api_key
        self.user_id = user_id
        self.storage_path = f"storage/workflows/{user_id}_n8n_workflows.json"
        
        # Workflow registry
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_executions: List[WorkflowExecution] = []
        self.persona_policies: Dict[str, Dict[str, Any]] = {}
        
        # Safety and monitoring
        self.safety_rules = self._initialize_safety_rules()
        self.monitoring_hooks: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_workflow_data()
        self._initialize_persona_policies()
    
    def _initialize_safety_rules(self) -> Dict[str, List[str]]:
        """Initialize safety rules for different workflow types"""
        return {
            'email_automation': [
                'never_delete_unread_important',
                'preserve_personal_relationships',
                'maintain_professional_tone',
                'respect_privacy_boundaries'
            ],
            'calendar_sync': [
                'preserve_personal_time',
                'respect_work_boundaries',
                'maintain_meeting_context',
                'prevent_double_booking'
            ],
            'file_organization': [
                'never_delete_without_backup',
                'preserve_file_timestamps',
                'maintain_folder_structure',
                'respect_access_permissions'
            ],
            'task_creation': [
                'respect_stress_levels',
                'maintain_work_life_balance',
                'preserve_task_context',
                'prevent_overwhelming_schedules'
            ],
            'social_media': [
                'maintain_authentic_voice',
                'respect_relationship_boundaries',
                'preserve_privacy_settings',
                'prevent_automated_responses_to_personal'
            ]
        }
    
    def _initialize_persona_policies(self):
        """Initialize persona-specific workflow policies"""
        self.persona_policies = {
            'mia': {
                'automation_philosophy': 'nurturing_supportive',
                'risk_tolerance': 'conservative',
                'intervention_style': 'gentle_guidance',
                'priority_values': ['emotional_wellbeing', 'relationship_preservation', 'stress_reduction'],
                'approval_required_for': ['email_deletion', 'calendar_changes', 'personal_data'],
                'monitoring_preferences': {
                    'emotional_impact_tracking': True,
                    'stress_level_monitoring': True,
                    'relationship_effect_analysis': True
                }
            },
            'solene': {
                'automation_philosophy': 'efficiency_focused',
                'risk_tolerance': 'moderate',
                'intervention_style': 'direct_optimization',
                'priority_values': ['productivity_maximization', 'goal_achievement', 'time_optimization'],
                'approval_required_for': ['major_calendar_restructuring', 'goal_modifications'],
                'monitoring_preferences': {
                    'productivity_metrics': True,
                    'goal_progress_tracking': True,
                    'efficiency_optimization': True
                }
            },
            'lyra': {
                'automation_philosophy': 'intuitive_harmonious',
                'risk_tolerance': 'moderate',
                'intervention_style': 'pattern_based_insights',
                'priority_values': ['creative_flow_preservation', 'pattern_recognition', 'holistic_balance'],
                'approval_required_for': ['creative_workflow_changes', 'inspiration_time_alterations'],
                'monitoring_preferences': {
                    'creative_pattern_tracking': True,
                    'flow_state_preservation': True,
                    'inspiration_timing_analysis': True
                }
            },
            'doc': {
                'automation_philosophy': 'systematic_analytical',
                'risk_tolerance': 'low',
                'intervention_style': 'data_driven_recommendations',
                'priority_values': ['data_integrity', 'systematic_organization', 'evidence_based_decisions'],
                'approval_required_for': ['data_modifications', 'system_changes', 'analysis_alterations'],
                'monitoring_preferences': {
                    'data_quality_tracking': True,
                    'system_performance_monitoring': True,
                    'accuracy_verification': True
                }
            }
        }
    
    async def create_workflow(self, workflow_config: Dict[str, Any], 
                            persona_name: str) -> Dict[str, Any]:
        """
        Create a new N8N workflow with persona oversight
        
        Args:
            workflow_config: N8N workflow configuration
            persona_name: Supervising persona
            
        Returns:
            Creation result with persona guidance
        """
        
        # Validate workflow safety
        safety_assessment = await self._assess_workflow_safety(workflow_config, persona_name)
        
        if not safety_assessment['approved']:
            return {
                'status': 'rejected',
                'reason': safety_assessment['reason'],
                'persona_feedback': safety_assessment['persona_feedback'],
                'suggestions': safety_assessment['suggestions']
            }
        
        # Enhance workflow with persona monitoring
        enhanced_config = await self._enhance_workflow_with_persona_oversight(
            workflow_config, persona_name, safety_assessment
        )
        
        # Create workflow in N8N
        try:
            n8n_response = await self._create_n8n_workflow(enhanced_config)
            
            if n8n_response['success']:
                workflow_id = n8n_response['workflow_id']
                
                # Register workflow locally
                self.active_workflows[workflow_id] = {
                    'config': enhanced_config,
                    'persona_supervisor': persona_name,
                    'safety_assessment': safety_assessment,
                    'created_at': datetime.now(),
                    'status': WorkflowStatus.APPROVED,
                    'monitoring_level': self._determine_monitoring_level(workflow_config, persona_name),
                    'execution_count': 0,
                    'last_execution': None
                }
                
                # Set up monitoring
                await self._setup_workflow_monitoring(workflow_id, persona_name)
                
                self._save_workflow_data()
                
                return {
                    'status': 'created',
                    'workflow_id': workflow_id,
                    'persona_guidance': self._generate_workflow_creation_guidance(
                        workflow_config, persona_name
                    ),
                    'monitoring_info': self._get_monitoring_info(workflow_id)
                }
            else:
                return {
                    'status': 'failed',
                    'reason': 'N8N creation failed',
                    'error': n8n_response.get('error', 'Unknown error')
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'reason': 'Technical error during creation',
                'error': str(e)
            }
    
    async def _assess_workflow_safety(self, workflow_config: Dict[str, Any], 
                                    persona_name: str) -> Dict[str, Any]:
        """Assess workflow safety with persona judgment"""
        
        workflow_type = workflow_config.get('type', 'unknown')
        triggers = workflow_config.get('triggers', [])
        actions = workflow_config.get('actions', [])
        
        # Get persona-specific safety policies
        persona_policy = self.persona_policies.get(persona_name, self.persona_policies['mia'])
        
        # Assess risk level
        risk_assessment = self._calculate_workflow_risk(workflow_config)
        
        # Check against safety rules
        safety_violations = self._check_safety_violations(workflow_config, workflow_type)
        
        # Persona-specific evaluation
        persona_evaluation = self._evaluate_with_persona_values(
            workflow_config, persona_policy, risk_assessment
        )
        
        # Determine approval
        approved = (
            len(safety_violations) == 0 and
            risk_assessment['overall_risk'] <= self._get_persona_risk_threshold(persona_name) and
            persona_evaluation['aligns_with_values']
        )
        
        return {
            'approved': approved,
            'risk_assessment': risk_assessment,
            'safety_violations': safety_violations,
            'persona_evaluation': persona_evaluation,
            'reason': self._generate_assessment_reason(approved, safety_violations, persona_evaluation),
            'persona_feedback': self._generate_persona_feedback(persona_evaluation, persona_name),
            'suggestions': self._generate_safety_suggestions(safety_violations, risk_assessment)
        }
    
    def _calculate_workflow_risk(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk level for workflow components"""
        
        risk_factors = {
            'data_modification': 0.0,
            'external_access': 0.0,
            'automation_scope': 0.0,
            'reversibility': 0.0,
            'privacy_impact': 0.0
        }
        
        actions = workflow_config.get('actions', [])
        
        for action in actions:
            action_type = action.get('type', '')
            
            # Data modification risk
            if action_type in ['delete', 'modify', 'move', 'archive']:
                risk_factors['data_modification'] += 0.3
            
            # External access risk
            if action_type in ['webhook', 'api_call', 'email_send', 'upload']:
                risk_factors['external_access'] += 0.2
            
            # Automation scope risk
            if action.get('batch_processing', False):
                risk_factors['automation_scope'] += 0.2
            
            # Reversibility risk
            if not action.get('reversible', True):
                risk_factors['reversibility'] += 0.4
            
            # Privacy risk
            if 'personal_data' in action.get('data_types', []):
                risk_factors['privacy_impact'] += 0.3
        
        # Calculate overall risk
        overall_risk = min(1.0, sum(risk_factors.values()) / len(risk_factors))
        
        return {
            'risk_factors': risk_factors,
            'overall_risk': overall_risk,
            'risk_level': 'low' if overall_risk < 0.3 else 'medium' if overall_risk < 0.7 else 'high'
        }
    
    def _evaluate_with_persona_values(self, workflow_config: Dict[str, Any], 
                                    persona_policy: Dict[str, Any], 
                                    risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate workflow against persona-specific values and priorities"""
        
        priority_values = persona_policy.get('priority_values', [])
        workflow_purpose = workflow_config.get('purpose', '')
        workflow_type = workflow_config.get('type', '')
        
        # Check alignment with persona values
        value_alignment_scores = {}
        
        for value in priority_values:
            if value == 'emotional_wellbeing':
                # Check if workflow reduces stress or supports emotional health
                score = 0.8 if 'stress_reduction' in workflow_purpose else 0.5
                if workflow_type in ['task_creation', 'calendar_sync']:
                    score += 0.2  # These can help with organization
            
            elif value == 'productivity_maximization':
                # Check if workflow enhances efficiency
                score = 0.9 if 'efficiency' in workflow_purpose else 0.6
                if workflow_type in ['email_automation', 'task_creation']:
                    score += 0.1
            
            elif value == 'creative_flow_preservation':
                # Check if workflow supports or disrupts creative work
                score = 0.8 if 'creative' in workflow_purpose else 0.5
                if workflow_type == 'notification_routing':
                    score += 0.2  # Can help manage interruptions
            
            elif value == 'data_integrity':
                # Check if workflow maintains data quality
                score = 0.9 if risk_assessment['risk_factors']['data_modification'] < 0.2 else 0.3
            
            else:
                score = 0.6  # Default neutral score
            
            value_alignment_scores[value] = min(1.0, score)
        
        # Calculate overall alignment
        overall_alignment = sum(value_alignment_scores.values()) / max(1, len(value_alignment_scores))
        
        # Check if approval required
        approval_required_for = persona_policy.get('approval_required_for', [])
        requires_approval = any(requirement in workflow_config.get('data_types', []) 
                              for requirement in approval_required_for)
        
        return {
            'aligns_with_values': overall_alignment >= 0.6,
            'value_alignment_scores': value_alignment_scores,
            'overall_alignment': overall_alignment,
            'requires_explicit_approval': requires_approval,
            'persona_recommendation': self._generate_persona_recommendation(
                overall_alignment, persona_policy['automation_philosophy']
            )
        }
    
    async def _enhance_workflow_with_persona_oversight(self, workflow_config: Dict[str, Any], 
                                                     persona_name: str, 
                                                     safety_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow configuration with persona-specific monitoring and guidance"""
        
        enhanced_config = workflow_config.copy()
        persona_policy = self.persona_policies.get(persona_name, self.persona_policies['mia'])
        
        # Add persona monitoring nodes
        monitoring_nodes = self._create_persona_monitoring_nodes(persona_name, workflow_config)
        enhanced_config['monitoring_nodes'] = monitoring_nodes
        
        # Add safety checkpoints
        safety_checkpoints = self._create_safety_checkpoints(safety_assessment, persona_policy)
        enhanced_config['safety_checkpoints'] = safety_checkpoints
        
        # Add persona guidance at key decision points
        guidance_points = self._create_persona_guidance_points(workflow_config, persona_name)
        enhanced_config['guidance_points'] = guidance_points
        
        # Add rollback mechanisms
        rollback_config = self._create_rollback_mechanisms(workflow_config, persona_policy)
        enhanced_config['rollback_config'] = rollback_config
        
        # Add emotional impact tracking
        if persona_policy['monitoring_preferences'].get('emotional_impact_tracking', False):
            enhanced_config['emotional_tracking'] = self._create_emotional_tracking_config(workflow_config)
        
        return enhanced_config
    
    def _create_persona_monitoring_nodes(self, persona_name: str, 
                                       workflow_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create monitoring nodes specific to persona oversight style"""
        
        monitoring_styles = {
            'mia': {
                'check_frequency': 'gentle_intervals',
                'focus_areas': ['emotional_impact', 'relationship_effects', 'stress_indicators'],
                'intervention_triggers': ['high_stress_detected', 'relationship_strain', 'overwhelm_signals']
            },
            'solene': {
                'check_frequency': 'efficiency_focused',
                'focus_areas': ['productivity_metrics', 'goal_progress', 'optimization_opportunities'],
                'intervention_triggers': ['efficiency_drops', 'goal_deviation', 'bottleneck_detection']
            },
            'lyra': {
                'check_frequency': 'intuitive_patterns',
                'focus_areas': ['creative_flow', 'pattern_recognition', 'holistic_balance'],
                'intervention_triggers': ['flow_disruption', 'pattern_anomalies', 'balance_shifts']
            },
            'doc': {
                'check_frequency': 'systematic_analysis',
                'focus_areas': ['data_quality', 'system_performance', 'accuracy_metrics'],
                'intervention_triggers': ['data_inconsistencies', 'performance_degradation', 'accuracy_issues']
            }
        }
        
        style = monitoring_styles.get(persona_name, monitoring_styles['mia'])
        
        monitoring_nodes = []
        for focus_area in style['focus_areas']:
            monitoring_nodes.append({
                'node_type': 'persona_monitor',
                'persona': persona_name,
                'focus_area': focus_area,
                'check_frequency': style['check_frequency'],
                'triggers': style['intervention_triggers']
            })
        
        return monitoring_nodes
    
    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow with persona oversight and monitoring
        
        Args:
            workflow_id: ID of workflow to execute
            trigger_data: Data that triggered the workflow
            
        Returns:
            Execution result with persona insights
        """
        
        if workflow_id not in self.active_workflows:
            return {'status': 'failed', 'reason': 'Workflow not found'}
        
        workflow_info = self.active_workflows[workflow_id]
        persona_supervisor = workflow_info['persona_supervisor']
        
        # Create execution record
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{workflow_id[:8]}"
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            start_time=datetime.now(),
            end_time=None,
            status=WorkflowStatus.RUNNING,
            persona_supervisor=persona_supervisor,
            results={},
            errors=[],
            persona_interventions=[]
        )
        
        self.workflow_executions.append(execution)
        
        try:
            # Pre-execution persona check
            pre_check = await self._persona_pre_execution_check(
                workflow_info, trigger_data, persona_supervisor
            )
            
            if not pre_check['approved']:
                execution.status = WorkflowStatus.PAUSED
                execution.persona_interventions.append({
                    'type': 'pre_execution_pause',
                    'reason': pre_check['reason'],
                    'timestamp': datetime.now(),
                    'persona_guidance': pre_check['guidance']
                })
                
                return {
                    'status': 'paused',
                    'execution_id': execution_id,
                    'reason': pre_check['reason'],
                    'persona_guidance': pre_check['guidance']
                }
            
            # Execute workflow in N8N
            n8n_result = await self._execute_n8n_workflow(workflow_id, trigger_data)
            
            # Monitor execution with persona oversight
            monitoring_result = await self._monitor_execution_with_persona(
                execution, n8n_result, persona_supervisor
            )
            
            # Complete execution
            execution.end_time = datetime.now()
            execution.status = WorkflowStatus.COMPLETED if n8n_result['success'] else WorkflowStatus.FAILED
            execution.results = n8n_result
            
            # Generate persona post-execution analysis
            post_analysis = await self._persona_post_execution_analysis(
                execution, workflow_info, persona_supervisor
            )
            
            # Update workflow statistics
            workflow_info['execution_count'] += 1
            workflow_info['last_execution'] = datetime.now()
            
            self._save_workflow_data()
            
            return {
                'status': 'completed' if n8n_result['success'] else 'failed',
                'execution_id': execution_id,
                'results': n8n_result,
                'persona_analysis': post_analysis,
                'monitoring_insights': monitoring_result
            }
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.errors.append(str(e))
            
            return {
                'status': 'failed',
                'execution_id': execution_id,
                'error': str(e)
            }
    
    async def _persona_pre_execution_check(self, workflow_info: Dict[str, Any], 
                                         trigger_data: Dict[str, Any], 
                                         persona_name: str) -> Dict[str, Any]:
        """Perform persona-specific pre-execution check"""
        
        persona_policy = self.persona_policies.get(persona_name, self.persona_policies['mia'])
        
        # Check current context
        current_context = await self._get_current_user_context()
        
        # Persona-specific pre-checks
        if persona_name == 'mia':
            # Check for high stress levels
            if current_context.get('stress_level', 0.5) > 0.7:
                return {
                    'approved': False,
                    'reason': 'High stress detected - protecting your well-being',
                    'guidance': 'I notice you might be feeling overwhelmed. Let\'s pause this automation and check in with you first.'
                }
        
        elif persona_name == 'solene':
            # Check for goal alignment
            if not self._workflow_aligns_with_current_goals(workflow_info, current_context):
                return {
                    'approved': False,
                    'reason': 'Workflow doesn\'t align with current priorities',
                    'guidance': 'This automation might distract from your key goals right now. Let\'s refocus on what matters most.'
                }
        
        elif persona_name == 'lyra':
            # Check creative flow state
            if current_context.get('in_flow_state', False):
                return {
                    'approved': False,
                    'reason': 'Preserving your creative flow',
                    'guidance': 'I sense you\'re in a beautiful creative flow. Let\'s not interrupt this precious state with automation.'
                }
        
        elif persona_name == 'doc':
            # Check data consistency
            data_quality_issues = await self._check_data_quality_for_workflow(workflow_info, trigger_data)
            if data_quality_issues:
                return {
                    'approved': False,
                    'reason': 'Data quality issues detected',
                    'guidance': f'Found {len(data_quality_issues)} data inconsistencies. Let\'s resolve these first to ensure accurate automation.'
                }
        
        return {
            'approved': True,
            'reason': 'Pre-execution checks passed',
            'guidance': f'{persona_name.title()} approves this workflow execution.'
        }
    
    # N8N API integration methods
    async def _create_n8n_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create workflow in N8N via API"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.n8n_base_url}/api/v1/workflows",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=workflow_config,
                    timeout=30
                )
                
                if response.status_code == 201:
                    result = response.json()
                    return {
                        'success': True,
                        'workflow_id': result.get('id'),
                        'data': result
                    }
                else:
                    return {
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
    
    async def _execute_n8n_workflow(self, workflow_id: str, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow in N8N"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.n8n_base_url}/api/v1/workflows/{workflow_id}/execute",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"triggerData": trigger_data},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'execution_data': result
                    }
                else:
                    return {
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
    
    # Helper methods for workflow management
    def _determine_monitoring_level(self, workflow_config: Dict[str, Any], persona_name: str) -> MonitoringLevel:
        """Determine appropriate monitoring level for workflow"""
        risk_assessment = self._calculate_workflow_risk(workflow_config)
        persona_policy = self.persona_policies.get(persona_name, self.persona_policies['mia'])
        
        if risk_assessment['overall_risk'] > 0.7:
            return MonitoringLevel.STRICT
        elif persona_policy['risk_tolerance'] == 'conservative':
            return MonitoringLevel.PERSONA_SUPERVISED
        elif risk_assessment['overall_risk'] > 0.4:
            return MonitoringLevel.STANDARD
        else:
            return MonitoringLevel.MINIMAL
    
    async def _setup_workflow_monitoring(self, workflow_id: str, persona_name: str):
        """Set up monitoring hooks for workflow"""
        monitoring_level = self._determine_monitoring_level(
            self.active_workflows[workflow_id]['config'], persona_name
        )
        
        # Create monitoring callbacks based on level
        if monitoring_level in [MonitoringLevel.STRICT, MonitoringLevel.PERSONA_SUPERVISED]:
            self.monitoring_hooks[workflow_id] = {
                'pre_execution': self._persona_pre_execution_check,
                'during_execution': self._monitor_execution_with_persona,
                'post_execution': self._persona_post_execution_analysis
            }
    
    def _generate_workflow_creation_guidance(self, workflow_config: Dict[str, Any], persona_name: str) -> str:
        """Generate persona-specific guidance for newly created workflow"""
        workflow_type = workflow_config.get('type', 'general')
        
        guidance_templates = {
            'mia': f"Your {workflow_type} workflow is ready! I'll watch over it gently, making sure it supports your well-being and doesn't add stress to your day.",
            'solene': f"Workflow activated for {workflow_type}. I'll monitor its performance and optimization opportunities to keep you focused on your goals.",
            'lyra': f"Your {workflow_type} automation is flowing now. I'll sense its rhythm and ensure it harmonizes with your natural patterns.",
            'doc': f"Workflow {workflow_type} initialized with systematic monitoring. Performance metrics and quality checks are in place."
        }
        
        return guidance_templates.get(persona_name, guidance_templates['mia'])
    
    def _get_monitoring_info(self, workflow_id: str) -> Dict[str, Any]:
        """Get monitoring information for workflow"""
        if workflow_id not in self.active_workflows:
            return {}
        
        workflow_info = self.active_workflows[workflow_id]
        return {
            'monitoring_level': workflow_info.get('monitoring_level', MonitoringLevel.STANDARD).value,
            'persona_supervisor': workflow_info.get('persona_supervisor', 'mia'),
            'execution_count': workflow_info.get('execution_count', 0),
            'last_execution': workflow_info.get('last_execution'),
            'safety_assessment': workflow_info.get('safety_assessment', {})
        }
    
    def _check_safety_violations(self, workflow_config: Dict[str, Any], workflow_type: str) -> List[str]:
        """Check for safety rule violations"""
        violations = []
        
        safety_rules = self.safety_rules.get(workflow_type, [])
        actions = workflow_config.get('actions', [])
        
        for action in actions:
            action_type = action.get('type', '')
            
            # Check specific violations
            if 'never_delete_unread_important' in safety_rules and action_type == 'delete':
                if not action.get('preserve_important', False):
                    violations.append("Workflow may delete important unread items")
            
            if 'preserve_personal_relationships' in safety_rules and action_type in ['email_send', 'message_send']:
                if action.get('auto_personal_responses', False):
                    violations.append("Automated personal responses detected")
            
            if 'never_delete_without_backup' in safety_rules and action_type == 'delete':
                if not action.get('backup_first', False):
                    violations.append("Deletion without backup protection")
        
        return violations
    
    def _get_persona_risk_threshold(self, persona_name: str) -> float:
        """Get risk tolerance threshold for persona"""
        thresholds = {
            'mia': 0.4,     # Conservative, protective
            'solene': 0.7,  # Moderate, goal-focused
            'lyra': 0.6,    # Moderate, pattern-aware
            'doc': 0.3      # Very conservative, data-focused
        }
        return thresholds.get(persona_name, 0.5)
    
    def _generate_assessment_reason(self, approved: bool, safety_violations: List[str], 
                                  persona_evaluation: Dict[str, Any]) -> str:
        """Generate human-readable reason for assessment result"""
        if approved:
            return "Workflow approved - meets safety and persona value requirements"
        
        reasons = []
        if safety_violations:
            reasons.append(f"Safety violations: {', '.join(safety_violations)}")
        
        if not persona_evaluation['aligns_with_values']:
            reasons.append(f"Doesn't align with persona values (score: {persona_evaluation['overall_alignment']:.2f})")
        
        return "; ".join(reasons) if reasons else "Assessment failed for unknown reasons"
    
    def _generate_persona_feedback(self, persona_evaluation: Dict[str, Any], persona_name: str) -> str:
        """Generate persona-specific feedback on workflow assessment"""
        alignment_score = persona_evaluation['overall_alignment']
        
        feedback_styles = {
            'mia': {
                'high': "This feels right for you - it supports your well-being and relationships beautifully.",
                'medium': "I have some gentle concerns about how this might affect your emotional balance.",
                'low': "This doesn't feel nurturing for you. Let's find a more caring approach."
            },
            'solene': {
                'high': "This workflow aligns perfectly with your goals. It's efficient and powerful.",
                'medium': "Good potential, but we can optimize this further for maximum impact.",
                'low': "This workflow won't drive you toward your goals effectively. Let's redesign it."
            },
            'lyra': {
                'high': "Beautiful flow - this automation harmonizes with your natural patterns perfectly.",
                'medium': "Interesting energy, but I sense some discord in the pattern. Let's adjust.",
                'low': "This disrupts your natural rhythm. We need something more in tune with your essence."
            },
            'doc': {
                'high': "Excellent alignment with systematic principles and data integrity requirements.",
                'medium': "Acceptable framework with some optimization opportunities identified.",
                'low': "Insufficient alignment with systematic methodology. Recommend restructuring."
            }
        }
        
        style = feedback_styles.get(persona_name, feedback_styles['mia'])
        
        if alignment_score >= 0.7:
            return style['high']
        elif alignment_score >= 0.4:
            return style['medium']
        else:
            return style['low']
    
    def _generate_safety_suggestions(self, safety_violations: List[str], 
                                   risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate suggestions to address safety concerns"""
        suggestions = []
        
        for violation in safety_violations:
            if "delete important" in violation:
                suggestions.append("Add importance checking before deletion actions")
            if "personal responses" in violation:
                suggestions.append("Remove automated responses to personal contacts")
            if "without backup" in violation:
                suggestions.append("Add backup step before any deletion operations")
        
        # Risk-based suggestions
        if risk_assessment['overall_risk'] > 0.6:
            suggestions.append("Consider adding more safety checkpoints")
            suggestions.append("Implement rollback mechanisms for high-risk actions")
        
        return suggestions
    
    def _generate_persona_recommendation(self, alignment_score: float, automation_philosophy: str) -> str:
        """Generate persona recommendation based on alignment and philosophy"""
        if alignment_score >= 0.7:
            return f"Strongly recommended - aligns with {automation_philosophy} approach"
        elif alignment_score >= 0.4:
            return f"Conditionally recommended with modifications for {automation_philosophy}"
        else:
            return f"Not recommended - conflicts with {automation_philosophy} values"
    
    def _create_safety_checkpoints(self, safety_assessment: Dict[str, Any], 
                                 persona_policy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create safety checkpoints for workflow"""
        checkpoints = []
        
        # Add checkpoints based on risk level
        risk_level = safety_assessment['risk_assessment']['risk_level']
        
        if risk_level in ['medium', 'high']:
            checkpoints.append({
                'type': 'data_verification',
                'description': 'Verify data integrity before processing',
                'required': True
            })
        
        if risk_level == 'high':
            checkpoints.append({
                'type': 'persona_approval',
                'description': 'Require persona approval for high-risk actions',
                'required': True
            })
        
        # Add persona-specific checkpoints
        if persona_policy['automation_philosophy'] == 'nurturing_supportive':
            checkpoints.append({
                'type': 'emotional_impact_check',
                'description': 'Assess emotional impact on user',
                'required': False
            })
        
        return checkpoints
    
    def _create_persona_guidance_points(self, workflow_config: Dict[str, Any], persona_name: str) -> List[Dict[str, Any]]:
        """Create persona guidance points throughout workflow"""
        guidance_points = []
        
        actions = workflow_config.get('actions', [])
        
        for i, action in enumerate(actions):
            if action.get('type') in ['delete', 'modify', 'send']:
                guidance_points.append({
                    'action_index': i,
                    'persona': persona_name,
                    'guidance_type': 'pre_action_check',
                    'message': self._get_pre_action_guidance(action, persona_name)
                })
        
        return guidance_points
    
    def _get_pre_action_guidance(self, action: Dict[str, Any], persona_name: str) -> str:
        """Get pre-action guidance message"""
        action_type = action.get('type', '')
        
        guidance_messages = {
            'mia': {
                'delete': "Gently checking if this deletion feels right for you...",
                'modify': "Making sure this change supports your well-being...",
                'send': "Ensuring this message reflects your caring nature..."
            },
            'solene': {
                'delete': "Verifying this deletion optimizes your workflow...",
                'modify': "Confirming this modification advances your goals...",
                'send': "Ensuring this message drives results..."
            },
            'lyra': {
                'delete': "Sensing if this deletion aligns with the natural flow...",
                'modify': "Checking if this change harmonizes with your patterns...",
                'send': "Ensuring this message carries the right energy..."
            },
            'doc': {
                'delete': "Systematically verifying deletion criteria...",
                'modify': "Analyzing modification impact on data integrity...",
                'send': "Validating message content and recipients..."
            }
        }
        
        persona_messages = guidance_messages.get(persona_name, guidance_messages['mia'])
        return persona_messages.get(action_type, f"Checking {action_type} action...")
    
    def _create_rollback_mechanisms(self, workflow_config: Dict[str, Any], 
                                  persona_policy: Dict[str, Any]) -> Dict[str, Any]:
        """Create rollback mechanisms for workflow"""
        return {
            'auto_rollback_enabled': persona_policy['risk_tolerance'] == 'conservative',
            'rollback_triggers': ['error_threshold_exceeded', 'persona_intervention', 'user_request'],
            'backup_retention_days': 30,
            'rollback_notification_required': True
        }
    
    def _create_emotional_tracking_config(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create emotional impact tracking configuration"""
        return {
            'track_stress_levels': True,
            'monitor_relationship_impact': True,
            'assess_workflow_satisfaction': True,
            'emotional_check_frequency': 'daily',
            'alert_thresholds': {
                'stress_increase': 0.2,
                'satisfaction_decrease': 0.3
            }
        }
    
    async def _monitor_execution_with_persona(self, execution: WorkflowExecution, 
                                            n8n_result: Dict[str, Any], 
                                            persona_name: str) -> Dict[str, Any]:
        """Monitor workflow execution with persona oversight"""
        monitoring_result = {
            'persona_observations': [],
            'intervention_needed': False,
            'emotional_impact_detected': False,
            'performance_metrics': {}
        }
        
        # Persona-specific monitoring
        if persona_name == 'mia':
            # Check for stress indicators
            if self._detect_stress_indicators(n8n_result):
                monitoring_result['emotional_impact_detected'] = True
                monitoring_result['persona_observations'].append(
                    "Detected potential stress impact - monitoring user well-being"
                )
        
        elif persona_name == 'solene':
            # Check performance metrics
            performance = self._analyze_workflow_performance(n8n_result)
            monitoring_result['performance_metrics'] = performance
            if performance.get('efficiency_score', 1.0) < 0.7:
                monitoring_result['intervention_needed'] = True
                monitoring_result['persona_observations'].append(
                    "Performance below optimal - optimization needed"
                )
        
        return monitoring_result
    
    async def _persona_post_execution_analysis(self, execution: WorkflowExecution, 
                                             workflow_info: Dict[str, Any], 
                                             persona_name: str) -> Dict[str, Any]:
        """Generate persona analysis after workflow execution"""
        return {
            'persona': persona_name,
            'execution_satisfaction': 0.8,  # Would be calculated based on actual metrics
            'recommendations': [
                f"Workflow executed successfully with {persona_name} oversight",
                "Consider minor optimizations for future runs"
            ],
            'learning_insights': [
                "User responded well to automated assistance",
                "No negative emotional impact detected"
            ]
        }
    
    async def _get_current_user_context(self) -> Dict[str, Any]:
        """Get current user context for decision making"""
        # Would integrate with other system components
        return {
            'stress_level': 0.4,
            'in_flow_state': False,
            'current_goals': ['productivity', 'well_being'],
            'available_time': 120  # minutes
        }
    
    def _workflow_aligns_with_current_goals(self, workflow_info: Dict[str, Any], 
                                          context: Dict[str, Any]) -> bool:
        """Check if workflow aligns with current user goals"""
        workflow_purpose = workflow_info['config'].get('purpose', '')
        current_goals = context.get('current_goals', [])
        
        # Simple alignment check
        for goal in current_goals:
            if goal in workflow_purpose.lower():
                return True
        
        return False
    
    async def _check_data_quality_for_workflow(self, workflow_info: Dict[str, Any], 
                                             trigger_data: Dict[str, Any]) -> List[str]:
        """Check data quality issues that might affect workflow"""
        issues = []
        
        # Check trigger data completeness
        required_fields = workflow_info['config'].get('required_trigger_fields', [])
        for field in required_fields:
            if field not in trigger_data or not trigger_data[field]:
                issues.append(f"Missing required field: {field}")
        
        return issues
    
    def _detect_stress_indicators(self, n8n_result: Dict[str, Any]) -> bool:
        """Detect stress indicators in workflow results"""
        # Simplified stress detection
        execution_time = n8n_result.get('execution_time_ms', 0)
        error_count = len(n8n_result.get('errors', []))
        
        return execution_time > 30000 or error_count > 2  # High execution time or errors
    
    def _analyze_workflow_performance(self, n8n_result: Dict[str, Any]) -> Dict[str, float]:
        """Analyze workflow performance metrics"""
        execution_time = n8n_result.get('execution_time_ms', 0)
        success_rate = 1.0 if n8n_result.get('success', False) else 0.0
        
        return {
            'execution_time_ms': execution_time,
            'success_rate': success_rate,
            'efficiency_score': min(1.0, 1.0 - (execution_time / 60000))  # Normalize to 1-minute baseline
        }

    # Data persistence methods
    def _save_workflow_data(self):
        """Save workflow data to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'active_workflows': {
                    k: {**v, 'created_at': v['created_at'].isoformat(),
                        'last_execution': v['last_execution'].isoformat() if v['last_execution'] else None}
                    for k, v in self.active_workflows.items()
                },
                'workflow_executions': [asdict(execution) for execution in self.workflow_executions[-100:]],
                'persona_policies': self.persona_policies
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"Error saving workflow data: {e}")
    
    def _load_workflow_data(self):
        """Load workflow data from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load active workflows
                if 'active_workflows' in data:
                    for workflow_id, workflow_data in data['active_workflows'].items():
                        workflow_data['created_at'] = datetime.fromisoformat(workflow_data['created_at'])
                        if workflow_data['last_execution']:
                            workflow_data['last_execution'] = datetime.fromisoformat(workflow_data['last_execution'])
                        self.active_workflows[workflow_id] = workflow_data
                
                # Load executions
                if 'workflow_executions' in data:
                    self.workflow_executions = [
                        WorkflowExecution(**exec_data) for exec_data in data['workflow_executions']
                    ]
                
            except Exception as e:
                print(f"Error loading workflow data: {e}")

# Factory and utility functions
def create_n8n_manager(n8n_base_url: str, api_key: str, user_id: str) -> N8NWorkflowManager:
    """Create an N8N workflow manager for a user"""
    return N8NWorkflowManager(n8n_base_url, api_key, user_id)

async def create_persona_supervised_workflow(n8n_base_url: str, api_key: str, user_id: str,
                                           workflow_config: Dict[str, Any], 
                                           persona_name: str) -> Dict[str, Any]:
    """Create a workflow with persona supervision"""
    manager = create_n8n_manager(n8n_base_url, api_key, user_id)
    return await manager.create_workflow(workflow_config, persona_name)
