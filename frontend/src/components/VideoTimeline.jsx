/**
 * Video Timeline Component
 * 
 * Interactive timeline scrubber for video analysis with frame-by-frame confidence display
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, SkipForward, SkipBack, Download } from 'lucide-react';

export default function VideoTimeline({
    frameResults = [],
    videoUrl,
    duration,
    onFrameSelect
}) {
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [hoveredFrame, setHoveredFrame] = useState(null);
    const timelineRef = useRef(null);
    const videoRef = useRef(null);

    const totalFrames = frameResults.length;

    // Auto-play through frames
    useEffect(() => {
        if (!isPlaying) return;

        const interval = setInterval(() => {
            setCurrentFrame(prev => {
                if (prev >= totalFrames - 1) {
                    setIsPlaying(false);
                    return prev;
                }
                return prev + 1;
            });
        }, 100); // 10 FPS playback

        return () => clearInterval(interval);
    }, [isPlaying, totalFrames]);

    // Sync video with current frame
    useEffect(() => {
        if (videoRef.current && frameResults[currentFrame]) {
            const timestamp = frameResults[currentFrame].timestamp;
            videoRef.current.currentTime = timestamp;
        }
    }, [currentFrame, frameResults]);

    const handleTimelineClick = (e) => {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = x / rect.width;
        const frameIndex = Math.floor(percentage * totalFrames);

        setCurrentFrame(Math.max(0, Math.min(frameIndex, totalFrames - 1)));
        if (onFrameSelect) {
            onFrameSelect(frameIndex);
        }
    };

    const getConfidenceColor = (confidence, prediction) => {
        if (prediction === 'fake') {
            return confidence > 0.8 ? '#ef4444' : confidence > 0.6 ? '#f59e0b' : '#fbbf24';
        } else {
            return confidence > 0.8 ? '#10b981' : confidence > 0.6 ? '#34d399' : '#6ee7b7';
        }
    };

    const currentResult = frameResults[currentFrame] || {};
    const hoveredResult = hoveredFrame !== null ? frameResults[hoveredFrame] : null;

    return (
        <div className="video-timeline-container">
            {/* Video Preview */}
            {videoUrl && (
                <div className="video-preview">
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        className="video-element"
                        muted
                    />

                    {/* Overlay Info */}
                    <div className="video-overlay">
                        <div className="frame-info">
                            <span className="frame-number">Frame {currentFrame + 1} / {totalFrames}</span>
                            <span className={`prediction ${currentResult.prediction}`}>
                                {currentResult.prediction?.toUpperCase()}
                            </span>
                            <span className="confidence">
                                {(currentResult.confidence * 100).toFixed(1)}% Confidence
                            </span>
                        </div>
                    </div>
                </div>
            )}

            {/* Timeline */}
            <div className="timeline-section">
                <div className="timeline-header">
                    <h3>Detection Timeline</h3>
                    <button className="export-btn" title="Export Timeline">
                        <Download size={18} />
                        Export
                    </button>
                </div>

                {/* Confidence Graph */}
                <div className="confidence-graph">
                    <svg width="100%" height="80" preserveAspectRatio="none">
                        {/* Background grid */}
                        <line x1="0" y1="20" x2="100%" y2="20" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                        <line x1="0" y1="40" x2="100%" y2="40" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                        <line x1="0" y1="60" x2="100%" y2="60" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

                        {/* Confidence line */}
                        <polyline
                            points={frameResults.map((result, i) => {
                                const x = (i / totalFrames) * 100;
                                const y = 80 - (result.confidence * 60);
                                return `${x}%,${y}`;
                            }).join(' ')}
                            fill="none"
                            stroke="url(#confidenceGradient)"
                            strokeWidth="2"
                            vectorEffect="non-scaling-stroke"
                        />

                        {/* Gradient definition */}
                        <defs>
                            <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                {frameResults.map((result, i) => (
                                    <stop
                                        key={i}
                                        offset={`${(i / totalFrames) * 100}%`}
                                        stopColor={getConfidenceColor(result.confidence, result.prediction)}
                                    />
                                ))}
                            </linearGradient>
                        </defs>

                        {/* Current position indicator */}
                        <line
                            x1={`${(currentFrame / totalFrames) * 100}%`}
                            y1="0"
                            x2={`${(currentFrame / totalFrames) * 100}%`}
                            y2="80"
                            stroke="#818cf8"
                            strokeWidth="2"
                        />
                    </svg>
                </div>

                {/* Frame Bars */}
                <div
                    ref={timelineRef}
                    className="frame-bars"
                    onClick={handleTimelineClick}
                >
                    {frameResults.map((result, index) => (
                        <motion.div
                            key={index}
                            className={`frame-bar ${index === currentFrame ? 'active' : ''}`}
                            style={{
                                backgroundColor: getConfidenceColor(result.confidence, result.prediction),
                                opacity: index === currentFrame ? 1 : 0.6
                            }}
                            onMouseEnter={() => setHoveredFrame(index)}
                            onMouseLeave={() => setHoveredFrame(null)}
                            whileHover={{ scaleY: 1.2 }}
                        />
                    ))}

                    {/* Hover Tooltip */}
                    {hoveredResult && hoveredFrame !== null && (
                        <motion.div
                            className="hover-tooltip"
                            style={{
                                left: `${(hoveredFrame / totalFrames) * 100}%`
                            }}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <div className="tooltip-content">
                                <div className="tooltip-frame">Frame {hoveredFrame + 1}</div>
                                <div className={`tooltip-prediction ${hoveredResult.prediction}`}>
                                    {hoveredResult.prediction}
                                </div>
                                <div className="tooltip-confidence">
                                    {(hoveredResult.confidence * 100).toFixed(1)}%
                                </div>
                            </div>
                        </motion.div>
                    )}
                </div>

                {/* Controls */}
                <div className="timeline-controls">
                    <button
                        className="control-btn"
                        onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}
                        disabled={currentFrame === 0}
                    >
                        <SkipBack size={20} />
                    </button>

                    <button
                        className="control-btn play-btn"
                        onClick={() => setIsPlaying(!isPlaying)}
                    >
                        {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                    </button>

                    <button
                        className="control-btn"
                        onClick={() => setCurrentFrame(Math.min(totalFrames - 1, currentFrame + 1))}
                        disabled={currentFrame === totalFrames - 1}
                    >
                        <SkipForward size={20} />
                    </button>

                    <div className="time-display">
                        {currentResult.timestamp?.toFixed(2)}s / {duration?.toFixed(2)}s
                    </div>
                </div>
            </div>

            <style jsx>{`
        .video-timeline-container {
          background: rgba(15, 23, 42, 0.6);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(99, 102, 241, 0.2);
          border-radius: 16px;
          padding: 24px;
          margin: 24px 0;
        }

        .video-preview {
          position: relative;
          width: 100%;
          max-width: 800px;
          margin: 0 auto 24px;
          border-radius: 12px;
          overflow: hidden;
          background: #000;
        }

        .video-element {
          width: 100%;
          display: block;
        }

        .video-overlay {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
          padding: 16px;
        }

        .frame-info {
          display: flex;
          gap: 16px;
          align-items: center;
          color: white;
          font-size: 14px;
        }

        .frame-number {
          font-weight: 600;
        }

        .prediction {
          padding: 4px 12px;
          border-radius: 6px;
          font-weight: 700;
          text-transform: uppercase;
          font-size: 12px;
        }

        .prediction.fake {
          background: rgba(239, 68, 68, 0.2);
          color: #fca5a5;
        }

        .prediction.real {
          background: rgba(16, 185, 129, 0.2);
          color: #6ee7b7;
        }

        .confidence {
          margin-left: auto;
          font-weight: 600;
          color: #a5b4fc;
        }

        .timeline-section {
          margin-top: 24px;
        }

        .timeline-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .timeline-header h3 {
          color: white;
          font-size: 18px;
          font-weight: 600;
          margin: 0;
        }

        .export-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          background: rgba(99, 102, 241, 0.1);
          border: 1px solid rgba(99, 102, 241, 0.3);
          border-radius: 8px;
          color: #a5b4fc;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .export-btn:hover {
          background: rgba(99, 102, 241, 0.2);
          border-color: rgba(99, 102, 241, 0.5);
        }

        .confidence-graph {
          margin-bottom: 12px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          padding: 8px;
        }

        .frame-bars {
          position: relative;
          display: flex;
          gap: 2px;
          height: 60px;
          cursor: pointer;
          margin-bottom: 16px;
        }

        .frame-bar {
          flex: 1;
          border-radius: 2px;
          transition: all 0.2s;
        }

        .frame-bar.active {
          box-shadow: 0 0 12px currentColor;
        }

        .hover-tooltip {
          position: absolute;
          bottom: 100%;
          transform: translateX(-50%);
          margin-bottom: 8px;
          pointer-events: none;
          z-index: 10;
        }

        .tooltip-content {
          background: rgba(15, 23, 42, 0.95);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(99, 102, 241, 0.3);
          border-radius: 8px;
          padding: 12px;
          white-space: nowrap;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .tooltip-frame {
          font-size: 12px;
          color: #94a3b8;
          margin-bottom: 4px;
        }

        .tooltip-prediction {
          font-size: 14px;
          font-weight: 700;
          text-transform: uppercase;
          margin-bottom: 4px;
        }

        .tooltip-prediction.fake {
          color: #fca5a5;
        }

        .tooltip-prediction.real {
          color: #6ee7b7;
        }

        .tooltip-confidence {
          font-size: 16px;
          font-weight: 600;
          color: #a5b4fc;
        }

        .timeline-controls {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
        }

        .control-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 40px;
          height: 40px;
          background: rgba(99, 102, 241, 0.1);
          border: 1px solid rgba(99, 102, 241, 0.3);
          border-radius: 50%;
          color: #a5b4fc;
          cursor: pointer;
          transition: all 0.2s;
        }

        .control-btn:hover:not(:disabled) {
          background: rgba(99, 102, 241, 0.2);
          border-color: rgba(99, 102, 241, 0.5);
          transform: scale(1.1);
        }

        .control-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .play-btn {
          width: 56px;
          height: 56px;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          border: none;
          color: white;
        }

        .play-btn:hover {
          transform: scale(1.1);
          box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
        }

        .time-display {
          margin-left: 16px;
          color: #94a3b8;
          font-size: 14px;
          font-weight: 600;
          font-variant-numeric: tabular-nums;
        }
      `}</style>
        </div>
    );
}
