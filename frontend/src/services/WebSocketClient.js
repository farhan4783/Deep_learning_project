/**
 * WebSocket Client for Real-time Detection Updates
 * 
 * Manages WebSocket connections and provides React hooks for easy integration
 */

class WebSocketClient {
    constructor(url, clientId) {
        this.url = url;
        this.clientId = clientId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.listeners = new Map();
        this.isConnecting = false;
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        if (this.isConnecting) {
            return new Promise((resolve) => {
                const checkConnection = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        clearInterval(checkConnection);
                        resolve();
                    }
                }, 100);
            });
        }

        this.isConnecting = true;

        return new Promise((resolve, reject) => {
            try {
                const wsUrl = `${this.url}/ws/${this.clientId}`;
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.isConnecting = false;
                    this.emit('connected');
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnecting = false;
                    this.emit('error', error);
                };

                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.isConnecting = false;
                    this.emit('disconnected');
                    this.attemptReconnect();
                };
            } catch (error) {
                this.isConnecting = false;
                reject(error);
            }
        });
    }

    handleMessage(data) {
        const { type } = data;

        // Emit event for specific message type
        this.emit(type, data);

        // Emit general message event
        this.emit('message', data);
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket not connected. Message not sent:', data);
        }
    }

    subscribeToTask(taskId) {
        this.send({
            type: 'subscribe',
            task_id: taskId
        });
    }

    unsubscribeFromTask(taskId) {
        this.send({
            type: 'unsubscribe',
            task_id: taskId
        });
    }

    ping() {
        this.send({ type: 'ping' });
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);

        // Return unsubscribe function
        return () => {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        };
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('reconnect_failed');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            this.connect().catch(error => {
                console.error('Reconnection failed:', error);
            });
        }, delay);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}

// React Hook for WebSocket
import { useEffect, useState, useCallback, useRef } from 'react';

export function useWebSocket(url, clientId) {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);

    useEffect(() => {
        // Create WebSocket client
        const ws = new WebSocketClient(url, clientId);
        wsRef.current = ws;

        // Set up event listeners
        ws.on('connected', () => setIsConnected(true));
        ws.on('disconnected', () => setIsConnected(false));
        ws.on('error', (err) => setError(err));
        ws.on('message', (data) => setLastMessage(data));

        // Connect
        ws.connect().catch(err => {
            console.error('Failed to connect:', err);
            setError(err);
        });

        // Cleanup on unmount
        return () => {
            ws.disconnect();
        };
    }, [url, clientId]);

    const send = useCallback((data) => {
        if (wsRef.current) {
            wsRef.current.send(data);
        }
    }, []);

    const subscribe = useCallback((taskId) => {
        if (wsRef.current) {
            wsRef.current.subscribeToTask(taskId);
        }
    }, []);

    const unsubscribe = useCallback((taskId) => {
        if (wsRef.current) {
            wsRef.current.unsubscribeFromTask(taskId);
        }
    }, []);

    const on = useCallback((event, callback) => {
        if (wsRef.current) {
            return wsRef.current.on(event, callback);
        }
    }, []);

    return {
        isConnected,
        lastMessage,
        error,
        send,
        subscribe,
        unsubscribe,
        on,
        client: wsRef.current
    };
}

// React Hook for Task Progress
export function useTaskProgress(taskId, wsClient) {
    const [progress, setProgress] = useState({
        stage: null,
        progress: 0,
        message: '',
        data: {}
    });
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (!wsClient) return;

        // Subscribe to task
        wsClient.subscribeToTask(taskId);

        // Listen for progress updates
        const unsubProgress = wsClient.on('progress', (data) => {
            if (data.task_id === taskId) {
                setProgress({
                    stage: data.stage,
                    progress: data.progress,
                    message: data.message,
                    data: data.data || {}
                });

                if (data.stage === 'complete') {
                    setIsComplete(true);
                }
            }
        });

        // Listen for results
        const unsubResult = wsClient.on('result', (data) => {
            if (data.task_id === taskId) {
                setResult(data.result);
                setIsComplete(true);
            }
        });

        // Listen for errors
        const unsubError = wsClient.on('error', (data) => {
            if (data.task_id === taskId) {
                setError(data.error);
                setIsComplete(true);
            }
        });

        // Cleanup
        return () => {
            wsClient.unsubscribeFromTask(taskId);
            if (unsubProgress) unsubProgress();
            if (unsubResult) unsubResult();
            if (unsubError) unsubError();
        };
    }, [taskId, wsClient]);

    return {
        progress,
        result,
        error,
        isComplete
    };
}

export default WebSocketClient;
