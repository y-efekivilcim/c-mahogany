import React, { useEffect, useRef, useState } from 'react';
import { GraphEngine } from './GraphEngine';
import './index.css';

function App() {
    const canvasRef = useRef(null);
    const engineRef = useRef(null);
    const [showMath, setShowMath] = useState(false);
    const [hasError, setHasError] = useState(false);

    const statusTextRef = useRef(null);
    const statusDotRef = useRef(null);
    const keRef = useRef(null);
    const tensionRef = useRef(null);
    const nodesRef = useRef(null);
    const edgesRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        let engine;
        let resizeObserver = null;
        
        document.fonts.ready.then(() => {
            engine = new GraphEngine(canvas);
            engineRef.current = engine;

            const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'https://mahogany-backend.onrender.com';
            const WS_URL = BACKEND_URL.replace(/^http/, 'ws');
            const ws = new WebSocket(`${WS_URL}/ws`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (engineRef.current) {
                    engineRef.current.updateFromBackend(data);
                }
            };

            const resize = () => {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
                if (engineRef.current) {
                    engineRef.current.width = canvas.width;
                    engineRef.current.height = canvas.height;
                    engineRef.current.init("mahogany");
                    
                    if (engineRef.current.ready) {
                        const targets = engineRef.current.nodes.map(n => [
                            n.targetX / canvas.width,
                            n.targetY / canvas.height
                        ]);
                        
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({ type: 'init', targets }));
                        }
                    }
                }
            };
            
            ws.onopen = () => {
                console.log('ws init sent');
                if (engineRef.current && engineRef.current.ready) {
                    const targets = engineRef.current.nodes.map(n => [
                        n.targetX / canvas.width,
                        n.targetY / canvas.height
                    ]);
                    ws.send(JSON.stringify({ type: 'init', targets }));
                }
            };
            
            window.addEventListener('resize', resize);
            resize();
            
            engine.onDisrupt = (disruptedNodes) => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'disrupt',
                        data: disruptedNodes
                    }));
                }
            };

            const onMove = (e) => { 
                engine.mouseX = e.clientX; 
                engine.mouseY = e.clientY; 
            };
            window.addEventListener('mousemove', onMove);
            
            const onTouch = (e) => {
                if (e.touches.length > 0) {
                    engine.mouseX = e.touches[0].clientX;
                    engine.mouseY = e.touches[0].clientY;
                }
            };
            const onTouchEnd = () => {
                engine.mouseX = -1000;
                engine.mouseY = -1000;
            };
            window.addEventListener('touchmove', onTouch);
            window.addEventListener('touchstart', onTouch);
            window.addEventListener('touchend', onTouchEnd);

            let rafId;
            let lastUpdate = 0;
            
            const loop = (timestamp) => {
                engine.step();
                engine.render();
                
                const isStable = engine.backendLoss !== undefined && 
                                 engine.backendLoss < 0.0001;
                                 
                if (statusTextRef.current && statusDotRef.current) {
                    let dotCount = Math.floor(timestamp / 400) % 4;
                    let dots = '.'.repeat(dotCount);
                    let paddedDots = dots.padEnd(3, ' ');
                    
                    if (engine.backendLoss === undefined) {
                        statusTextRef.current.innerText = `WAKING UP BACKEND${paddedDots}`;
                        statusTextRef.current.style.color = 'var(--accent-dim)';
                        statusDotRef.current.style.backgroundColor = 'var(--accent-dim)';
                        statusDotRef.current.style.boxShadow = '0 0 10px var(--accent-dim)';
                    } else if (isStable) {
                        statusTextRef.current.innerText = 'NETWORK CONVERGED   ';
                        statusTextRef.current.style.color = 'var(--accent-glow)';
                        statusDotRef.current.style.backgroundColor = 'var(--accent-glow)';
                        statusDotRef.current.style.boxShadow = '0 0 10px var(--accent-glow)';
                    } else {
                        statusTextRef.current.innerText = `TRAINING MANIFOLD${paddedDots}`;
                        statusTextRef.current.style.color = 'var(--accent-dim)';
                        statusDotRef.current.style.backgroundColor = 'var(--accent-dim)';
                        statusDotRef.current.style.boxShadow = '0 0 10px var(--accent-dim)';
                    }
                }
                
                if (engine.showMath && (timestamp - lastUpdate > 100)) {
                    lastUpdate = timestamp;
                    if (keRef.current) {
                        keRef.current.innerText = engine.backendLoss !== undefined ? engine.backendLoss.toFixed(5) : '0.00000';
                    }
                    if (tensionRef.current) {
                        tensionRef.current.innerText = engine.backendGradNorm !== undefined ? engine.backendGradNorm.toFixed(5) : '0.00000';
                    }
                    
                    if (nodesRef.current) {
                        nodesRef.current.innerText = engine.nodes.length;
                    }
                    if (edgesRef.current) {
                        edgesRef.current.innerText = engine.edges.length;
                    }
                }
                
                rafId = requestAnimationFrame(loop);
            };
            rafId = requestAnimationFrame(loop);

            return () => {
                window.removeEventListener('resize', resize);
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('touchmove', onTouch);
                window.removeEventListener('touchstart', onTouch);
                window.removeEventListener('touchend', onTouchEnd);
                cancelAnimationFrame(rafId);
                ws.close();
            };
        });
    }, []);

    useEffect(() => {
        if (engineRef.current) {
            engineRef.current.showMath = showMath;
        }
    }, [showMath]);

    return (
        <div className={`app-container ${showMath ? 'math-mode' : ''}`}>
            <canvas ref={canvasRef} className="gl-canvas" />
            
            <div className="hero-overlay">
                <header className="hero-header" style={{maxWidth: '500px'}}>
                    <p>A live PyTorch simulation. Watch a neural network physically self-organize its latent coordinates in real-time.</p>
                </header>
                
                {showMath && (
                    <div className="glass-panel">
                        <div className="glass-header">PYTORCH LATENT OPTIMIZATION</div>
                        
                        <div className="glass-section">
                            <div className="glass-label">OPTIMIZER ALGORITHM</div>
                            <div className="glass-value text-white">Stochastic Gradient Descent</div>
                            <div className="glass-subtext">LR: 0.01 · Momentum (β): 0.0</div>
                        </div>

                        <div className="glass-section">
                            <div className="glass-label">LOSS FUNCTION (MSE)</div>
                            <div className="glass-value highlight-cyan" ref={keRef}>0.00000</div>
                        </div>

                        <div className="glass-section">
                            <div className="glass-label">GRADIENT NORM (||∇W||)</div>
                            <div className="glass-value highlight-magenta" ref={tensionRef}>0.00000</div>
                        </div>

                        <div className="glass-grid">
                            <div className="glass-stat">
                                <div className="glass-label">ACTIVE NODES</div>
                                <div className="glass-value small" ref={nodesRef}>0</div>
                            </div>
                            <div className="glass-stat">
                                <div className="glass-label">TOPOLOGY EDGES</div>
                                <div className="glass-value small" ref={edgesRef}>0</div>
                            </div>
                        </div>

                        <div className="glass-divider"></div>

                        <div className="glass-legend">
                            <div className="legend-row"><span className="glow-dot cyan"></span> TARGET VECTOR (L2)</div>
                            <div className="legend-row"><span className="glow-dot magenta"></span> KINETIC VELOCITY</div>
                            <div className="legend-row"><span className="glow-dot purple"></span> SPRING TOPOLOGY</div>
                        </div>
                    </div>
                )}
                
                <div className="bottom-bar">
                    <div className="status-indicator">
                        <div ref={statusDotRef} className="dot loading"></div>
                        <span ref={statusTextRef}>CALCULATING MANIFOLD</span>
                    </div>
                    
                    <button 
                        className={`math-toggle ${showMath ? 'active math-mode-btn' : ''}`}
                        onClick={() => setShowMath(!showMath)}
                    >
                        {showMath ? 'DISABLE MATH' : 'VIEW MATH'}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default App;
