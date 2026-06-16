export class Node {
    constructor(x, y, id) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.targetX = x;
        this.targetY = y;
    }
}

export class Edge {
    constructor(a, b, restLength) {
        this.a = a;
        this.b = b;
        this.restLength = restLength;
    }
}

export class GraphEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        
        this.nodes = [];
        this.edges = [];
        this.showMath = false;
        
        this.mouseX = -1000;
        this.mouseY = -1000;
        
        this.backendLoss = undefined;
        this.backendGradNorm = undefined;
        this.backendPreds = null;
        
        this.kineticEnergy = 0;
        this.ready = false;
    }
    
    updateFromBackend(data) {
        this.backendLoss = data.loss;
        this.backendGradNorm = data.grad_norm;
        this.backendPreds = data.preds;
    }
    
    init(text = "mahogany") {
        this.nodes = [];
        this.edges = [];
        
        const offCanvas = document.createElement('canvas');
        offCanvas.width = this.width;
        offCanvas.height = this.height;
        const octx = offCanvas.getContext('2d', { willReadFrequently: true });
        
        octx.fillStyle = '#000';
        octx.fillRect(0, 0, this.width, this.height);
        octx.fillStyle = '#fff';
        
        const fontSize = Math.min(this.width * 0.18, 306);
        octx.font = `400 ${fontSize}px 'Great Vibes', cursive`;
        octx.textAlign = 'center';
        octx.textBaseline = 'middle';
        octx.fillText(text, this.width / 2, this.height / 2);
        
        const imgData = octx.getImageData(0, 0, this.width, this.height).data;
        const targets = [];
        const step = 7;
        
        for (let y = 0; y < this.height; y += step) {
            for (let x = 0; x < this.width; x += step) {
                const idx = (y * this.width + x) * 4;
                if (imgData[idx] > 128) {
                    targets.push({ x, y });
                }
            }
        }
        
        const cx = this.width / 2;
        const cy = this.height / 2;
        for (let i = 0; i < targets.length; i++) {
            const startX = cx + (Math.random() - 0.5) * 50;
            const startY = cy + (Math.random() - 0.5) * 50;
            const n = new Node(startX, startY, i);
            n.targetX = targets[i].x;
            n.targetY = targets[i].y;
            this.nodes.push(n);
        }
        
        const grid = {};
        const cellSize = 30;
        for (const n of this.nodes) {
            const gx = Math.floor(n.targetX / cellSize);
            const gy = Math.floor(n.targetY / cellSize);
            const key = `${gx},${gy}`;
            if (!grid[key]) grid[key] = [];
            grid[key].push(n);
        }
        
        const connectRadius = 25;
        for (const n1 of this.nodes) {
            let connections = 0;
            const gx = Math.floor(n1.targetX / cellSize);
            const gy = Math.floor(n1.targetY / cellSize);
            
            for (let x = gx - 1; x <= gx + 1; x++) {
                for (let y = gy - 1; y <= gy + 1; y++) {
                    const cell = grid[`${x},${y}`];
                    if (!cell) continue;
                    
                    for (const n2 of cell) {
                        if (n1 === n2 || connections >= 3) continue;
                        const dx = n1.targetX - n2.targetX;
                        const dy = n1.targetY - n2.targetY;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        
                        if (dist < connectRadius) {
                            let exists = false;
                            for (const e of this.edges) {
                                if ((e.a === n1 && e.b === n2) || (e.a === n2 && e.b === n1)) {
                                    exists = true;
                                    break;
                                }
                            }
                            if (!exists) {
                                this.edges.push(new Edge(n1, n2, dist));
                                connections++;
                            }
                        }
                    }
                }
            }
        }
        
        this.ready = true;
    }
    
    step() {
        if (this.ready === false) return;
        this.kineticEnergy = 0;
        
        for (let i = 0; i < this.edges.length; i++) {
            const e = this.edges[i];
            const dx = e.b.x - e.a.x;
            const dy = e.b.y - e.a.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist > 0) {
                const diff = (dist - e.restLength) / dist;
                const force = diff * 0.1;
                e.a.vx += dx * force;
                e.a.vy += dy * force;
                e.b.vx -= dx * force;
                e.b.vy -= dy * force;
            }
        }
        
        const disruptedNodes = [];
        for (let i = 0; i < this.nodes.length; i++) {
            const n = this.nodes[i];
            let disrupted = false;
            
            const dxM = n.x - this.mouseX;
            const dyM = n.y - this.mouseY;
            const distSq = dxM*dxM + dyM*dyM;
            if (!this.showMath && distSq < 8000) {
                const f = 150 / (distSq + 1);
                n.vx += (dxM / Math.sqrt(distSq)) * f;
                n.vy += (dyM / Math.sqrt(distSq)) * f;
                disrupted = true;
            }
            
            if (this.backendPreds && this.backendPreds.length > i) {
                const pred = this.backendPreds[i];
                const px = pred[0] * this.width;
                const py = pred[1] * this.height;
                
                n.vx += (px - n.x) * 0.15;
                n.vy += (py - n.y) * 0.15;
            } else {
                n.vx += (this.width/2 - n.x) * 0.02;
                n.vy += (this.height/2 - n.y) * 0.02;
            }
            
            n.vx *= 0.78;
            n.vy *= 0.78;
            n.x += n.vx;
            n.y += n.vy;
            
            this.kineticEnergy += (n.vx*n.vx + n.vy*n.vy);
            
            if (disrupted) {
                disruptedNodes.push([i, n.x / this.width, n.y / this.height]);
            }
        }
        
        if (disruptedNodes.length > 0 && this.onDisrupt) {
            this.onDisrupt(disruptedNodes);
        }
    }
    
    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);
        
        if (!this.ready) return;
        
        if (this.showMath) {
            this.renderMathView(ctx);
        } else {
            this.renderBeautyView(ctx);
        }
    }
    
    renderBeautyView(ctx) {
        ctx.strokeStyle = 'rgba(74, 21, 21, 0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < this.edges.length; i++) {
            const e = this.edges[i];
            ctx.moveTo(e.a.x, e.a.y);
            ctx.lineTo(e.b.x, e.b.y);
        }
        ctx.stroke();
        
        ctx.fillStyle = '#4a1515';
        for (let i = 0; i < this.nodes.length; i++) {
            const n = this.nodes[i];
            ctx.beginPath();
            ctx.arc(n.x, n.y, 2.0, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    renderMathView(ctx) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for(let x=0; x<this.width; x+=50) { ctx.moveTo(x, 0); ctx.lineTo(x, this.height); }
        for(let y=0; y<this.height; y+=50) { ctx.moveTo(0, y); ctx.lineTo(this.width, y); }
        ctx.stroke();
        
        ctx.strokeStyle = 'rgba(209, 0, 255, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < this.edges.length; i++) {
            const e = this.edges[i];
            ctx.moveTo(e.a.x, e.a.y);
            ctx.lineTo(e.b.x, e.b.y);
        }
        ctx.stroke();
        
        ctx.fillStyle = '#ffffff';
        for (let i = 0; i < this.nodes.length; i++) {
            const n = this.nodes[i];
            ctx.fillRect(n.x - 2.5, n.y - 2.5, 5, 5);
            
            if (this.backendPreds && this.backendPreds.length > i) {
                const px = this.backendPreds[i][0] * this.width;
                const py = this.backendPreds[i][1] * this.height;
                ctx.strokeStyle = 'rgba(0, 150, 255, 0.8)';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(n.x, n.y);
                ctx.lineTo(px, py);
                ctx.stroke();
            }
            
            ctx.strokeStyle = 'rgba(255, 51, 102, 1.0)';
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.moveTo(n.x, n.y);
            ctx.lineTo(n.x + n.vx * 15, n.y + n.vy * 15);
            ctx.stroke();
        }
        
        let closestNode = null;
        let minDistSq = Infinity;
        for (let i = 0; i < this.nodes.length; i++) {
            const n = this.nodes[i];
            const dx = n.x - this.mouseX;
            const dy = n.y - this.mouseY;
            const distSq = dx*dx + dy*dy;
            if (distSq < minDistSq) {
                minDistSq = distSq;
                closestNode = n;
            }
        }
        
        if (closestNode && minDistSq < 10000) {
            const n = closestNode;
            
            ctx.strokeStyle = '#00ffff';
            ctx.shadowColor = '#00ffff';
            ctx.shadowBlur = 10;
            ctx.lineWidth = 1;
            ctx.strokeRect(n.x - 8, n.y - 8, 16, 16);
            ctx.shadowBlur = 0;
            
            ctx.fillStyle = 'rgba(5, 5, 5, 0.8)';
            ctx.fillRect(n.x + 15, n.y + 15, 180, 55);
            ctx.strokeStyle = '#00ffff';
            ctx.strokeRect(n.x + 15, n.y + 15, 180, 55);
            
            ctx.fillStyle = '#00ffff';
            ctx.font = '10px "Space Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`ID: ${n.id}`, n.x + 20, n.y + 30);
            ctx.fillText(`P : [${n.x.toFixed(1)}, ${n.y.toFixed(1)}]`, n.x + 20, n.y + 45);
            ctx.fillText(`V : [${n.vx.toFixed(2)}, ${n.vy.toFixed(2)}]`, n.x + 20, n.y + 60);
        }
    }
}
