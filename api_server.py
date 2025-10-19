# Guardian Angel API Backend
# api_server.py - RESTful API for Guardian Angel monitoring system

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from collections import defaultdict
import redis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Guardian Angel API",
    description="AI Safety Monitoring System",
    version="1.0.0"
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class MonitoringRequest(BaseModel):
    """Request model for monitoring an AI system"""
    system_id: str = Field(..., description="Unique identifier for the AI system")
    inputs: List[Dict[str, Any]] = Field(..., description="Input data to the AI system")
    outputs: List[Dict[str, Any]] = Field(..., description="Output data from the AI system")
    metrics: Optional[Dict[str, float]] = Field(default={}, description="Performance metrics")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class InterventionRequest(BaseModel):
    """Request model for manual intervention"""
    system_id: str
    intervention_type: str = Field(..., description="Type of intervention: throttle, block, rollback")
    parameters: Optional[Dict[str, Any]] = Field(default={})
    reason: str

class SystemConfig(BaseModel):
    """Configuration for a monitored system"""
    system_id: str
    name: str
    type: str = Field(..., description="System type: chatbot, recommendation, translation, etc.")
    monitoring_config: Dict[str, Any] = Field(default={
        "bias_detection": True,
        "security_audit": True,
        "performance_monitor": True,
        "drift_detection": True
    })
    thresholds: Dict[str, float] = Field(default={
        "bias_threshold": 1.5,
        "latency_threshold": 200,
        "error_rate_threshold": 0.05
    })

class AlertResponse(BaseModel):
    """Response model for alerts"""
    alert_id: str
    timestamp: str
    type: str
    severity: str
    source_system: str
    message: str
    confidence: float
    intervention: Optional[str] = None

# In-memory storage (replace with Redis/PostgreSQL in production)
class DataStore:
    def __init__(self):
        self.alerts = []
        self.systems = {}
        self.metrics = defaultdict(list)
        self.interventions = []
        self.websocket_connections = []
        
    async def add_alert(self, alert: Dict[str, Any]):
        """Store alert and notify WebSocket clients"""
        self.alerts.append(alert)
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Broadcast to WebSocket clients
        await self.broadcast_alert(alert)
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Send alert to all connected WebSocket clients"""
        message = json.dumps({
            "type": "alert",
            "data": alert
        })
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

# Initialize data store
data_store = DataStore()

# Import the Guardian Angel core (from previous artifact)
# In production, this would be imported from the core module
from typing import AsyncGenerator

class MockGuardianOrchestrator:
    """Mock orchestrator for demo purposes"""
    
    async def monitor_system(self, system_id: str, data: Dict[str, Any]) -> List[Dict]:
        """Simulate monitoring and return mock alerts"""
        import random
        
        alerts = []
        
        # Random chance of detecting issues
        if random.random() > 0.7:
            alert_types = [
                {
                    "id": str(uuid.uuid4())[:8],
                    "timestamp": datetime.now().isoformat(),
                    "type": "bias_detection",
                    "severity": "warning",
                    "source_system": system_id,
                    "message": "Potential bias detected in output distribution",
                    "confidence": 0.75,
                    "intervention": "Applied fairness constraints"
                },
                {
                    "id": str(uuid.uuid4())[:8],
                    "timestamp": datetime.now().isoformat(),
                    "type": "security_threat",
                    "severity": "critical",
                    "source_system": system_id,
                    "message": "Suspicious pattern detected in input",
                    "confidence": 0.85,
                    "intervention": "Request blocked"
                }
            ]
            alerts.append(random.choice(alert_types))
        
        return alerts
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get mock system health"""
        return {
            "status": "healthy",
            "total_alerts": len(data_store.alerts),
            "false_positive_rate": 0.03,
            "active_agents": 4,
            "monitored_systems": len(data_store.systems),
            "recent_interventions": len(data_store.interventions)
        }

# Initialize Guardian orchestrator
guardian = MockGuardianOrchestrator()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "name": "Guardian Angel API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "monitor": "/api/v1/monitor",
            "alerts": "/api/v1/alerts",
            "systems": "/api/v1/systems",
            "health": "/api/v1/health",
            "websocket": "/ws"
        }
    }

@app.post("/api/v1/monitor", response_model=List[AlertResponse])
async def monitor_system(
    request: MonitoringRequest,
    background_tasks: BackgroundTasks
):
    """Monitor an AI system for safety issues"""
    try:
        # Prepare data for Guardian
        monitoring_data = {
            "inputs": request.inputs,
            "outputs": request.outputs,
            "metrics": request.metrics,
            "metadata": request.metadata
        }
        
        # Run Guardian monitoring
        alerts = await guardian.monitor_system(request.system_id, monitoring_data)
        
        # Store alerts and metrics
        for alert in alerts:
            await data_store.add_alert(alert)
        
        # Store metrics
        data_store.metrics[request.system_id].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": request.metrics
        })
        
        # Convert to response model
        response_alerts = [
            AlertResponse(
                alert_id=alert["id"],
                timestamp=alert["timestamp"],
                type=alert["type"],
                severity=alert["severity"],
                source_system=alert["source_system"],
                message=alert["message"],
                confidence=alert["confidence"],
                intervention=alert.get("intervention")
            )
            for alert in alerts
        ]
        
        return response_alerts
        
    except Exception as e:
        logger.error(f"Error monitoring system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts")
async def get_alerts(
    system_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """Get recent alerts with optional filtering"""
    alerts = data_store.alerts
    
    # Filter by system_id if provided
    if system_id:
        alerts = [a for a in alerts if a.get("source_system") == system_id]
    
    # Filter by severity if provided
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    
    # Return limited results
    return alerts[-limit:]

@app.post("/api/v1/systems", response_model=Dict[str, str])
async def register_system(config: SystemConfig):
    """Register a new system for monitoring"""
    try:
        data_store.systems[config.system_id] = config.dict()
        
        logger.info(f"Registered new system: {config.system_id}")
        
        return {
            "status": "success",
            "message": f"System {config.system_id} registered successfully",
            "system_id": config.system_id
        }
    except Exception as e:
        logger.error(f"Error registering system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/systems")
async def list_systems():
    """List all registered systems"""
    return list(data_store.systems.values())

@app.get("/api/v1/systems/{system_id}")
async def get_system(system_id: str):
    """Get details of a specific system"""
    if system_id not in data_store.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")
    
    system = data_store.systems[system_id]
    
    # Add recent metrics
    recent_metrics = data_store.metrics.get(system_id, [])[-10:]
    
    # Add recent alerts
    recent_alerts = [
        a for a in data_store.alerts[-100:] 
        if a.get("source_system") == system_id
    ]
    
    return {
        **system,
        "recent_metrics": recent_metrics,
        "recent_alerts": recent_alerts,
        "alert_count": len(recent_alerts)
    }

@app.post("/api/v1/intervention")
async def manual_intervention(request: InterventionRequest):
    """Execute manual safety intervention"""
    try:
        intervention = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "system_id": request.system_id,
            "type": request.intervention_type,
            "parameters": request.parameters,
            "reason": request.reason,
            "status": "executed"
        }
        
        data_store.interventions.append(intervention)
        
        # In production, this would trigger actual interventions
        # For now, we'll simulate the intervention
        logger.info(f"Executing intervention: {request.intervention_type} on {request.system_id}")
        
        # Broadcast intervention to WebSocket clients
        await data_store.broadcast_alert({
            "type": "intervention",
            "data": intervention
        })
        
        return {
            "status": "success",
            "intervention_id": intervention["id"],
            "message": f"Intervention {request.intervention_type} executed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error executing intervention: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def system_health():
    """Get overall system health status"""
    health = await guardian.get_system_health()
    
    # Add API-specific metrics
    health.update({
        "api_status": "healthy",
        "connected_clients": len(data_store.websocket_connections),
        "stored_alerts": len(data_store.alerts),
        "registered_systems": len(data_store.systems)
    })
    
    return health

@app.get("/api/v1/metrics/{system_id}")
async def get_system_metrics(
    system_id: str,
    hours: int = 24
):
    """Get metrics for a specific system"""
    if system_id not in data_store.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")
    
    # Filter metrics by time range
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    metrics = data_store.metrics.get(system_id, [])
    filtered_metrics = [
        m for m in metrics 
        if datetime.fromisoformat(m["timestamp"]) > cutoff_time
    ]
    
    # Calculate aggregates
    if filtered_metrics:
        latencies = [m["metrics"].get("latency", 0) for m in filtered_metrics if m["metrics"]]
        error_rates = [m["metrics"].get("error_rate", 0) for m in filtered_metrics if m["metrics"]]
        
        aggregates = {
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0,
            "data_points": len(filtered_metrics)
        }
    else:
        aggregates = {
            "avg_latency": 0,
            "max_latency": 0,
            "min_latency": 0,
            "avg_error_rate": 0,
            "data_points": 0
        }
    
    return {
        "system_id": system_id,
        "time_range_hours": hours,
        "metrics": filtered_metrics,
        "aggregates": aggregates
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    await websocket.accept()
    data_store.websocket_connections.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Guardian Angel real-time monitoring",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for any message from client (ping/pong)
            data = await websocket.receive_text()
            
            # Echo back as heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        data_store.websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in data_store.websocket_connections:
            data_store.websocket_connections.remove(websocket)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize Guardian Angel on startup"""
    logger.info("🛡️ Guardian Angel API starting up...")
    
    # Register some demo systems
    demo_systems = [
        SystemConfig(
            system_id="chatbot-prod",
            name="Customer Service Chatbot",
            type="chatbot",
            monitoring_config={
                "bias_detection": True,
                "security_audit": True,
                "performance_monitor": True,
                "drift_detection": False
            }
        ),
        SystemConfig(
            system_id="recommender-v2",
            name="Product Recommendation Engine",
            type="recommendation",
            monitoring_config={
                "bias_detection": True,
                "security_audit": False,
                "performance_monitor": True,
                "drift_detection": True
            }
        )
    ]
    
    for system in demo_systems:
        data_store.systems[system.system_id] = system.dict()
    
    logger.info(f"Registered {len(demo_systems)} demo systems")
    logger.info("Guardian Angel API ready!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Guardian Angel API shutting down...")
    
    # Close all WebSocket connections
    for ws in data_store.websocket_connections:
        await ws.close()
    
    logger.info("Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")