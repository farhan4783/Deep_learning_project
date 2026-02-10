"""
WebSocket API for Real-time Deepfake Detection Updates

This module provides WebSocket endpoints for live progress tracking
during detection processing.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
from datetime import datetime


router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates
    """
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_connections: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Accept new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        """
        Remove WebSocket connection
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from task connections
        for task_id in list(self.task_connections.keys()):
            if client_id in self.task_connections[task_id]:
                self.task_connections[task_id].remove(client_id)
                if not self.task_connections[task_id]:
                    del self.task_connections[task_id]
    
    def subscribe_to_task(self, client_id: str, task_id: str):
        """
        Subscribe client to task updates
        
        Args:
            client_id: Client identifier
            task_id: Task identifier
        """
        if task_id not in self.task_connections:
            self.task_connections[task_id] = set()
        self.task_connections[task_id].add(client_id)
    
    async def send_personal_message(self, message: dict, client_id: str):
        """
        Send message to specific client
        
        Args:
            message: Message dictionary
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_task(self, message: dict, task_id: str):
        """
        Broadcast message to all clients subscribed to a task
        
        Args:
            message: Message dictionary
            task_id: Task identifier
        """
        if task_id in self.task_connections:
            disconnected_clients = []
            
            for client_id in self.task_connections[task_id]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_json(message)
                    except Exception as e:
                        print(f"Error broadcasting to {client_id}: {e}")
                        disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    async def send_progress_update(
        self,
        task_id: str,
        stage: str,
        progress: float,
        message: str,
        data: dict = None
    ):
        """
        Send progress update for a task
        
        Args:
            task_id: Task identifier
            stage: Current processing stage
            progress: Progress percentage (0-100)
            message: Status message
            data: Additional data
        """
        update = {
            'type': 'progress',
            'task_id': task_id,
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        await self.broadcast_to_task(update, task_id)
    
    async def send_result(
        self,
        task_id: str,
        result: dict,
        success: bool = True
    ):
        """
        Send final result for a task
        
        Args:
            task_id: Task identifier
            result: Result data
            success: Whether task completed successfully
        """
        message = {
            'type': 'result',
            'task_id': task_id,
            'success': success,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_to_task(message, task_id)
    
    async def send_error(
        self,
        task_id: str,
        error: str,
        details: dict = None
    ):
        """
        Send error message for a task
        
        Args:
            task_id: Task identifier
            error: Error message
            details: Additional error details
        """
        message = {
            'type': 'error',
            'task_id': task_id,
            'error': error,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_to_task(message, task_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                # Subscribe to task updates
                task_id = data.get('task_id')
                if task_id:
                    manager.subscribe_to_task(client_id, task_id)
                    await manager.send_personal_message({
                        'type': 'subscribed',
                        'task_id': task_id,
                        'message': f'Subscribed to task {task_id}'
                    }, client_id)
            
            elif message_type == 'ping':
                # Respond to ping
                await manager.send_personal_message({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }, client_id)
            
            elif message_type == 'unsubscribe':
                # Unsubscribe from task
                task_id = data.get('task_id')
                if task_id and task_id in manager.task_connections:
                    if client_id in manager.task_connections[task_id]:
                        manager.task_connections[task_id].remove(client_id)
                        await manager.send_personal_message({
                            'type': 'unsubscribed',
                            'task_id': task_id
                        }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")
    
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)


async def send_detection_progress(
    task_id: str,
    stage: str,
    progress: float,
    message: str,
    **kwargs
):
    """
    Helper function to send detection progress updates
    
    Args:
        task_id: Task identifier
        stage: Processing stage ('upload', 'preprocessing', 'detection', 'explanation')
        progress: Progress percentage (0-100)
        message: Status message
        **kwargs: Additional data to include
    """
    await manager.send_progress_update(
        task_id=task_id,
        stage=stage,
        progress=progress,
        message=message,
        data=kwargs
    )


async def send_frame_result(
    task_id: str,
    frame_idx: int,
    total_frames: int,
    confidence: float,
    prediction: str
):
    """
    Send result for individual video frame
    
    Args:
        task_id: Task identifier
        frame_idx: Frame index
        total_frames: Total number of frames
        confidence: Detection confidence
        prediction: Prediction label
    """
    progress = (frame_idx + 1) / total_frames * 100
    
    await manager.send_progress_update(
        task_id=task_id,
        stage='detection',
        progress=progress,
        message=f'Processing frame {frame_idx + 1}/{total_frames}',
        data={
            'frame_idx': frame_idx,
            'confidence': confidence,
            'prediction': prediction
        }
    )


# Export manager for use in other modules
__all__ = ['router', 'manager', 'send_detection_progress', 'send_frame_result']
