/**
 * 3D Confidence Visualization Component
 * 
 * Interactive 3D bar chart showing model confidence scores using Three.js
 */

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';

function ConfidenceBar({ position, height, color, label, confidence }) {
    const meshRef = useRef();
    const [hovered, setHovered] = React.useState(false);

    useFrame((state) => {
        if (meshRef.current) {
            // Gentle floating animation
            meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.05;

            // Scale on hover
            const targetScale = hovered ? 1.1 : 1;
            meshRef.current.scale.x = THREE.MathUtils.lerp(meshRef.current.scale.x, targetScale, 0.1);
            meshRef.current.scale.z = THREE.MathUtils.lerp(meshRef.current.scale.z, targetScale, 0.1);
        }
    });

    return (
        <group position={position}>
            {/* Bar */}
            <mesh
                ref={meshRef}
                position={[0, height / 2, 0]}
                onPointerOver={() => setHovered(true)}
                onPointerOut={() => setHovered(false)}
            >
                <boxGeometry args={[0.8, height, 0.8]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={hovered ? 0.5 : 0.2}
                    metalness={0.5}
                    roughness={0.2}
                />
            </mesh>

            {/* Label */}
            <Text
                position={[0, -0.5, 0]}
                fontSize={0.3}
                color="white"
                anchorX="center"
                anchorY="middle"
            >
                {label}
            </Text>

            {/* Confidence value */}
            <Text
                position={[0, height + 0.5, 0]}
                fontSize={0.25}
                color={color}
                anchorX="center"
                anchorY="middle"
                font="/fonts/Inter-Bold.ttf"
            >
                {(confidence * 100).toFixed(1)}%
            </Text>

            {/* Tooltip on hover */}
            {hovered && (
                <Html position={[0, height + 1, 0]} center>
                    <div className="tooltip-3d">
                        <div className="tooltip-label">{label}</div>
                        <div className="tooltip-value">{(confidence * 100).toFixed(2)}%</div>
                    </div>
                </Html>
            )}
        </group>
    );
}

function Particles() {
    const particlesRef = useRef();
    const particleCount = 100;

    const positions = useMemo(() => {
        const pos = new Float32Array(particleCount * 3);
        for (let i = 0; i < particleCount; i++) {
            pos[i * 3] = (Math.random() - 0.5) * 20;
            pos[i * 3 + 1] = (Math.random() - 0.5) * 20;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
        }
        return pos;
    }, []);

    useFrame((state) => {
        if (particlesRef.current) {
            particlesRef.current.rotation.y = state.clock.elapsedTime * 0.05;
        }
    });

    return (
        <points ref={particlesRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={particleCount}
                    array={positions}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial
                size={0.05}
                color="#4f46e5"
                transparent
                opacity={0.6}
                sizeAttenuation
            />
        </points>
    );
}

function Scene({ modelScores }) {
    const groupRef = useRef();

    useFrame((state) => {
        if (groupRef.current) {
            groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
        }
    });

    const colors = {
        'EfficientNet': '#10b981',
        'XceptionNet': '#3b82f6',
        'Custom CNN': '#8b5cf6',
        'ViT': '#f59e0b',
        'Ensemble': '#ef4444'
    };

    return (
        <>
            {/* Ambient lighting */}
            <ambientLight intensity={0.5} />

            {/* Directional lights */}
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <directionalLight position={[-10, -10, -5]} intensity={0.5} />

            {/* Point light for dramatic effect */}
            <pointLight position={[0, 5, 0]} intensity={1} color="#4f46e5" />

            {/* Bars */}
            <group ref={groupRef}>
                {Object.entries(modelScores).map(([model, score], index) => {
                    const x = (index - (Object.keys(modelScores).length - 1) / 2) * 2;
                    const height = score * 5;

                    return (
                        <ConfidenceBar
                            key={model}
                            position={[x, 0, 0]}
                            height={height}
                            color={colors[model] || '#6366f1'}
                            label={model}
                            confidence={score}
                        />
                    );
                })}
            </group>

            {/* Particles */}
            <Particles />

            {/* Grid floor */}
            <gridHelper args={[20, 20, '#4f46e5', '#1e1b4b']} position={[0, -0.5, 0]} />

            {/* Controls */}
            <OrbitControls
                enablePan={false}
                enableZoom={true}
                minDistance={5}
                maxDistance={15}
                maxPolarAngle={Math.PI / 2}
            />
        </>
    );
}

export default function ThreeVisualization({ modelScores }) {
    return (
        <div className="three-visualization-container">
            <Canvas
                camera={{ position: [0, 5, 10], fov: 50 }}
                style={{ background: 'linear-gradient(to bottom, #0f172a, #1e1b4b)' }}
            >
                <Scene modelScores={modelScores} />
            </Canvas>

            <style jsx>{`
        .three-visualization-container {
          width: 100%;
          height: 500px;
          border-radius: 16px;
          overflow: hidden;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        :global(.tooltip-3d) {
          background: rgba(15, 23, 42, 0.95);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(99, 102, 241, 0.3);
          border-radius: 8px;
          padding: 12px 16px;
          color: white;
          font-family: 'Inter', sans-serif;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
          pointer-events: none;
        }

        :global(.tooltip-label) {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 4px;
          color: #a5b4fc;
        }

        :global(.tooltip-value) {
          font-size: 20px;
          font-weight: 700;
          color: #818cf8;
        }
      `}</style>
        </div>
    );
}
