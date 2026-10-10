import React, { useMemo, useState, useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Center, Sparkles, Line, Environment, ContactShadows, MeshReflectorMaterial, Lightformer, Html } from '@react-three/drei';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

const MATERIALS = {
    default:  { color: '#06b6d4', metalness: 0.85, roughness: 0.25 },
    titanium: { color: '#8a8f98', metalness: 0.95, roughness: 0.22 },
    aluminum: { color: '#c9cdd4', metalness: 1.0,  roughness: 0.30 },
    steel:    { color: '#b8bcc4', metalness: 1.0,  roughness: 0.28 },
    brass:    { color: '#c99a2e', metalness: 1.0,  roughness: 0.22 },
    copper:   { color: '#c97f4a', metalness: 1.0,  roughness: 0.25 },
    carbon:   { color: '#23252a', metalness: 0.55, roughness: 0.45 },
    glass:    { color: '#bfe8f5', metalness: 0.0,  roughness: 0.05, transparent: true, opacity: 0.55, transmission: 1.0 },
    rubber:   { color: '#2b2b2e', metalness: 0.0,  roughness: 0.92 },
    plastic:  { color: '#e6e7e9', metalness: 0.05, roughness: 0.45 },
    wood:     { color: '#8a5a2b', metalness: 0.0,  roughness: 0.70 },
    ceramic:  { color: '#f2f0ea', metalness: 0.0,  roughness: 0.35 },
};

// --- Materials ---
const GeometryModel = ({ geometry, material, hologram, selected }) => {
    const selRef = useRef();
    // Selection glow pulse (Stark targeting)
    useFrame((state) => {
        if (selRef.current) {
            selRef.current.emissiveIntensity = 0.45 + Math.sin(state.clock.elapsedTime * 5) * 0.25;
        }
    });
    if (hologram) {
        return (
            <group>
                <mesh geometry={geometry} castShadow receiveShadow>
                    <meshStandardMaterial color="#22d3ee" transparent opacity={0.22} emissive="#0ea5e9" emissiveIntensity={0.35} metalness={0.2} roughness={0.3} depthWrite={false} />
                </mesh>
                <mesh geometry={geometry} scale={[1.002, 1.002, 1.002]}>
                    <meshBasicMaterial color="#22d3ee" wireframe transparent opacity={0.28} blending={THREE.AdditiveBlending} depthWrite={false} />
                </mesh>
            </group>
        );
    }
    return (
        <mesh geometry={geometry} castShadow receiveShadow>
            {material.transmission ? (
                <meshPhysicalMaterial ref={selected ? selRef : null} color={material.color} metalness={material.metalness} roughness={material.roughness} transparent opacity={material.opacity} transmission={material.transmission} ior={1.5} envMapIntensity={1.6} clearcoat={0.4} emissive={selected ? '#22d3ee' : '#000'} />
            ) : (
                <meshPhysicalMaterial
                    ref={selected ? selRef : null}
                    color={material.color}
                    metalness={material.metalness}
                    roughness={material.roughness}
                    envMapIntensity={1.5}
                    clearcoat={material.metalness > 0.5 ? 0.55 : 0.15}
                    clearcoatRoughness={0.28}
                    reflectivity={material.metalness > 0.5 ? 0.9 : 0.5}
                    emissive={selected ? '#0891b2' : '#000'}
                />
            )}
        </mesh>
    );
};

const HologramScan = ({ size }) => {
    const ref = useRef();
    useFrame((state) => {
        if (!ref.current) return;
        const t = state.clock.elapsedTime;
        ref.current.position.y = (Math.sin(t * 0.8) * 0.5 + 0.5) * size.y - size.y / 2;
    });
    return (
        <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[size.x * 1.6, size.z * 1.6]} />
            <meshBasicMaterial color="#22d3ee" transparent opacity={0.12} blending={THREE.AdditiveBlending} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
    );
};

const LoadingCube = () => {
    const meshRef = React.useRef();
    useFrame((state, delta) => {
        meshRef.current.rotation.x += delta;
        meshRef.current.rotation.y += delta;
    });
    return (
        <mesh ref={meshRef}>
            <boxGeometry args={[10, 10, 10]} />
            <meshStandardMaterial wireframe color="cyan" transparent opacity={0.5} />
        </mesh>
    );
};

// --- Scene rig -----------------------------------------------------------------
// Everything the viewer used to hard-code (camera distance, lights, floor, fog, shadow
// map) was tuned for one size of model. STLs arrive in real millimetres, so a 10 mm cube
// framed fine while a 120 mm part put the camera *inside* the mesh. The rig derives all of
// it from the model's own bounding box, which is what makes any scale look intentional.
const VIEW_FOV = 38;
function computeRig(dims) {
    if (!dims) return null;
    const { size } = dims;
    const maxDim = Math.max(size.x, size.y, size.z) || 10;
    const radius = Math.max(Math.hypot(size.x, size.y, size.z) / 2, 1);
    const groundY = -size.y / 2;                 // model is centred, so its base sits here
    const pad = Math.max(maxDim * 0.14, 2.5);
    const dist = (radius / Math.sin((VIEW_FOV * Math.PI) / 360)) * 1.5;
    return { maxDim, radius, groundY, pad, dist };
}

const CameraRig = ({ rig }) => {
    const { camera } = useThree();
    useEffect(() => {
        if (!rig) return;
        const dir = new THREE.Vector3(1, 0.6, 1.15).normalize();
        camera.position.copy(dir.multiplyScalar(rig.dist));
        camera.near = Math.max(rig.dist * 0.005, 0.01);
        camera.far = rig.dist * 50;
        camera.updateProjectionMatrix();
        camera.lookAt(0, 0, 0);
    }, [rig, camera]);
    return null;
};

// --- Measurement annotations (the engineering-drawing look) ---------------------
const DimChip = ({ children, tone = 'cyan' }) => (
    <div
        style={{ pointerEvents: 'none', userSelect: 'none' }}
        className={`px-1.5 py-0.5 rounded border backdrop-blur-sm text-[10px] font-mono tracking-wider whitespace-nowrap shadow-[0_0_10px_rgba(34,211,238,0.25)] ${tone === 'amber' ? 'border-amber-400/40 bg-black/70 text-amber-200' : 'border-cyan-400/40 bg-black/70 text-cyan-200'}`}
    >
        {children}
    </div>
);

const DimTick = ({ at, along }) => (
    <Line
        points={[at.clone().sub(along), at.clone().add(along)]}
        color="#67e8f9"
        lineWidth={1}
        transparent
        opacity={0.8}
    />
);

const Dimension = ({ from, to, tickDir, tickLen, label }) => {
    const mid = from.clone().add(to).multiplyScalar(0.5);
    const along = tickDir.clone().normalize().multiplyScalar(tickLen);
    return (
        <group>
            <Line points={[from, to]} color="#67e8f9" lineWidth={1.2} transparent opacity={0.85} />
            <DimTick at={from} along={along} />
            <DimTick at={to} along={along} />
            <Html position={mid} center zIndexRange={[5, 0]} style={{ pointerEvents: 'none' }}>
                <DimChip>{label}</DimChip>
            </Html>
        </group>
    );
};

const ExtensionLine = ({ from, to }) => (
    <Line
        points={[from, to]}
        color="#22d3ee"
        lineWidth={0.6}
        transparent
        opacity={0.28}
        dashed
        dashScale={4}
        dashSize={1}
        gapSize={1.3}
    />
);

const DimensionOverlay = ({ dims, rig }) => {
    const dimsSpec = useMemo(() => {
        const { size } = dims;
        const { pad, groundY } = rig;
        const hx = size.x / 2, hz = size.z / 2;
        const tickLen = pad * 0.42;
        const yLine = groundY - pad * 0.55;       // dimension lines sit below the part
        const zOut = hz + pad * 0.7;
        const xOut = hx + pad * 0.7;
        const xH = -hx - pad * 0.7;
        const V = (x, y, z) => new THREE.Vector3(x, y, z);
        return {
            tickLen,
            width: {
                from: V(-hx, yLine, zOut), to: V(hx, yLine, zOut), tickDir: V(0, 1, 0),
                label: `W ${size.x.toFixed(1)} mm`,
                ext: [[V(-hx, groundY, zOut), V(-hx, yLine, zOut)], [V(hx, groundY, zOut), V(hx, yLine, zOut)]],
            },
            depth: {
                from: V(xOut, yLine, -hz), to: V(xOut, yLine, hz), tickDir: V(0, 1, 0),
                label: `D ${size.z.toFixed(1)} mm`,
                ext: [[V(xOut, groundY, -hz), V(xOut, yLine, -hz)], [V(xOut, groundY, hz), V(xOut, yLine, hz)]],
            },
            height: {
                from: V(xH, groundY, zOut), to: V(xH, groundY + size.y, zOut), tickDir: V(1, 0, 0),
                label: `H ${size.y.toFixed(1)} mm`,
                ext: [[V(xH, groundY, zOut), V(-hx, groundY, hz)], [V(xH, groundY + size.y, zOut), V(-hx, groundY + size.y, hz)]],
            },
        };
    }, [dims, rig]);

    return (
        <group>
            {['width', 'depth', 'height'].map((k) => {
                const d = dimsSpec[k];
                return (
                    <group key={k}>
                        {d.ext.map((e, i) => <ExtensionLine key={i} from={e[0]} to={e[1]} />)}
                        <Dimension from={d.from} to={d.to} tickDir={d.tickDir} tickLen={dimsSpec.tickLen} label={d.label} />
                    </group>
                );
            })}
        </group>
    );
};

// --- Hologram HUD ---------------------------------------------------------------
const ScannerRing = ({ rig }) => {
    const arc = useRef();
    const ring = useRef();
    useFrame((state) => {
        const t = state.clock.elapsedTime;
        if (arc.current) arc.current.rotation.z = t * 0.8;
        if (ring.current) ring.current.rotation.z = -t * 0.32;
    });
    const r = rig.radius * 1.25;
    const lift = rig.groundY + rig.maxDim * 0.015;
    return (
        <group position={[0, lift, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <mesh ref={ring}>
                <ringGeometry args={[r, r + rig.maxDim * 0.006, 96]} />
                <meshBasicMaterial color="#22d3ee" transparent opacity={0.35} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} depthWrite={false} />
            </mesh>
            <mesh ref={arc}>
                <ringGeometry args={[r * 1.1, r * 1.1 + rig.maxDim * 0.02, 64, 1, 0, Math.PI * 1.1]} />
                <meshBasicMaterial color="#67e8f9" transparent opacity={0.6} side={THREE.DoubleSide} blending={THREE.AdditiveBlending} depthWrite={false} />
            </mesh>
        </group>
    );
};

const HoloReadout = ({ name, dims, rig, material }) => (
    <Html
        position={new THREE.Vector3(0, rig.groundY + dims.size.y + rig.pad * 0.95, 0)}
        center
        zIndexRange={[5, 0]}
        style={{ pointerEvents: 'none' }}
    >
        <div className="flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-md border border-cyan-400/40 bg-cyan-950/40 backdrop-blur-sm shadow-[0_0_18px_rgba(34,211,238,0.3)]">
            <div className="text-[11px] font-mono tracking-[0.2em] text-cyan-100 uppercase">{name || 'design'}</div>
            <div className="text-[10px] font-mono text-cyan-300/80">
                {dims.size.x.toFixed(1)} × {dims.size.y.toFixed(1)} × {dims.size.z.toFixed(1)} mm · {(material || 'metal').toUpperCase()}
            </div>
        </div>
    </Html>
);

// --- Explode helpers ---
function inferAxis(size) {
    // size is Vector3
    const dims = [size.x, size.y, size.z];
    const axes = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
    const sorted = dims.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v);
    const small = sorted[0], large = sorted[2], mid = sorted[1];
    // Plate: one thin dimension < 0.5 * mid
    if (small.v < mid.v * 0.55) {
        return axes[small.i].clone(); // explode along thin axis (normal to plate)
    }
    // Rod: one long dimension > 2 * mid
    if (large.v > mid.v * 1.9) {
        return axes[large.i].clone(); // explode along long axis (shaft length)
    }
    // Otherwise radial — will be handled as outward, fallback to Z
    return new THREE.Vector3(0,0,1);
}

function easeInOutCubic(t) {
    return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2, 3)/2;
}

// Trail line between assembled and exploded
const ExplodeTrail = ({ start, end, opacity }) => {
    if (opacity <= 0.02) return null;
    return (
        <Line
            points={[start, end]}
            color="#22d3ee"
            lineWidth={1.2}
            dashed
            dashScale={6}
            dashSize={2}
            gapSize={1.5}
            transparent
            opacity={opacity * 0.85}
        />
    );
};

const ExplodedPart = ({ part, index, total, material, hologram, dims, explodeFactor, animatedFactor, showTrails, showGhost, spacing, assemblyAxis, simResult, selected, onPick, label }) => {
    // Per-part material (Stark-lab realism) with global fallback
    const partMaterial = useMemo(() => MATERIALS[part.materialKey] || material, [part.materialKey, material]);
    // Prefer backend-provided axis (perfect), otherwise infer from geometry
    const axis = useMemo(() => {
        if (assemblyAxis) return new THREE.Vector3(...assemblyAxis);
        part.geometry.computeBoundingBox();
        const sz = new THREE.Vector3().subVectors(part.geometry.boundingBox.max, part.geometry.boundingBox.min);
        return inferAxis(sz);
    }, [part.geometry, assemblyAxis]);

    // Smart distance: bigger parts move proportionally farther
    const partSize = useMemo(() => {
        part.geometry.computeBoundingBox();
        const sz = new THREE.Vector3().subVectors(part.geometry.boundingBox.max, part.geometry.boundingBox.min);
        return Math.max(sz.x, sz.y, sz.z);
    }, [part.geometry]);

    // Where the part actually sits, so a hologram label can track it
    const partCenter = useMemo(() => {
        part.geometry.computeBoundingBox();
        return part.geometry.boundingBox.getCenter(new THREE.Vector3());
    }, [part.geometry]);

    // Alternate direction along axis for symmetric explode (like shaft out one way, gear the other)
    // For 2 parts: one +axis, one -axis. For N, distribute.
    const direction = useMemo(() => {
        if (total === 1) return new THREE.Vector3(0,0,0);
        // Alternate sign: even index +, odd -
        // For more than 2, spread along axis with slight radial offset to avoid overlap
        const sign = index % 2 === 0 ? 1 : -1;
        // For 3+ parts, add a small radial component so they don't all sit on same line
        // Keep primary along inferred axis
        const v = axis.clone().multiplyScalar(sign);
        if (total > 2) {
            // Add a tiny orthogonal offset for visual separation
            const offset = (index / total - 0.5) * 0.3;
            v.x += offset * (axis.z !== 0 ? 1 : 0);
            v.y += offset * (axis.x !== 0 ? 1 : 0);
            v.normalize();
        }
        return v;
    }, [axis, index, total]);

    // Smart distance: base + part size factor, so larger parts travel farther (like real CAD)
    const distance = useMemo(() => {
        const base = spacing;
        const sizeFactor = 1 + (partSize / (dims ? Math.max(dims.size.x, dims.size.y, dims.size.z) : 50)) * 0.6;
        return base * sizeFactor;
    }, [spacing, partSize, dims]);

    // Sequential easing: each part starts 0.13 after previous (suit-assembly effect)
    const effective = useMemo(() => {
        const delay = index * 0.13;
        const denom = Math.max(0.01, 1 - delay);
        const t = Math.min(1, Math.max(0, (animatedFactor - delay) / denom));
        return easeInOutCubic(t);
    }, [animatedFactor, index]);

    const offset = direction.clone().multiplyScalar(distance * effective * explodeFactor);

    const trailEnd = offset.clone();
    const trailOpacity = effective * explodeFactor;

    // Heatmap tint based on FEA — template-free real physics
    const heatMaterial = (() => {
        if (!simResult || !simResult.status) return material;
        if (simResult.status === 'PASS') return { color: '#22c55e', metalness: 0.1, roughness: 0.4 };
        if (simResult.status === 'FAIL') return { color: '#ef4444', metalness: 0.1, roughness: 0.4 };
        if (simResult.status === 'WARN') return { color: '#eab308', metalness: 0.1, roughness: 0.4 };
        return material;
    })();

    return (
        <group>
            {showGhost && explodeFactor > 0.05 && (
                <mesh geometry={part.geometry} position={[0,0,0]}>
                    <meshStandardMaterial color={material.color} transparent opacity={0.08} depthWrite={false} />
                </mesh>
            )}
            {showTrails && part.geometry && (
                <ExplodeTrail start={new THREE.Vector3(0,0,0)} end={trailEnd} opacity={trailOpacity} />
            )}
            <group position={offset}>
                <GeometryModel geometry={part.geometry} material={simResult ? heatMaterial : partMaterial} hologram={hologram} selected={selected}
                    onPointerDown={(e) => { e.stopPropagation(); onPick?._(part.name, e); }} />
                {hologram && dims && <HologramScan size={dims.size} />}
                {hologram && label && (
                    <Html position={partCenter.clone().add(new THREE.Vector3(0, partSize * 0.62 + 2, 0))} center zIndexRange={[5, 0]} style={{ pointerEvents: 'none' }}>
                        <div className="px-2 py-0.5 rounded border border-cyan-400/50 bg-black/70 backdrop-blur-sm text-[10px] font-mono tracking-wider text-cyan-200 whitespace-nowrap shadow-[0_0_12px_rgba(34,211,238,0.3)]">
                            {label}
                        </div>
                    </Html>
                )}
            </group>
        </group>
    );
};

const CadWindow = ({ data, thoughts, retryInfo = {}, onClose, socket }) => {
    const [isIterating, setIsIterating] = useState(false);
    const [prompt, setPrompt] = useState("");
    const [isSending, setIsSending] = useState(false);
    const [holoMode, setHoloMode] = useState(false);
    const [showDims, setShowDims] = useState(true);
    const [explodeFactor, setExplodeFactor] = useState(0); // 0 assembled, 1 fully exploded
    const [showTrails, setShowTrails] = useState(true);
    const [showGhost, setShowGhost] = useState(true);
    const [animatedFactor, setAnimatedFactor] = useState(0);
    const [simulating, setSimulating] = useState(false);
    const [simResult, setSimResult] = useState(null); // {max_stress, safety_factor, status}
    const [editState, setEditState] = useState({});   // {partId: {visible, color, scale}}
    const [selectedPart, setSelectedPart] = useState(null);
    const [showTeam, setShowTeam] = useState(false); // Level 3: plan + findings panel
    const pickDownRef = useRef(null);
    const thoughtsEndRef = useRef(null);

    // Live model surgery: server pushes edit events (delete/hide/isolate/recolor/rescale/undo)
    useEffect(() => {
        if (!socket) return;
        const handler = (event) => {
            if (!event || !event.visibility) return;
            const next = {};
            event.visibility.forEach((v) => {
                next[v.id] = { visible: v.visible !== false, color: v.color || null, scale: v.scale || 1 };
            });
            setEditState(next);
        };
        socket.on('model_edit', handler);
        return () => socket.off('model_edit', handler);
    }, [socket]);

    // Point & Speak: clicking a part makes it "THIS" for the next command
    const handlePartPick = (partName, e) => {
        if (pickDownRef.current && e.button === 0) {
            const dx = e.clientX - pickDownRef.current.x;
            const dy = e.clientY - pickDownRef.current.y;
            if (Math.hypot(dx, dy) > 6) return; // was an orbit-drag, not a click
        }
        setSelectedPart(partName);
        socket?.emit('part_pick', { name: partName });
        console.log('[SURGERY] Picked:', partName);
    };

    // Clear edits when a NEW model arrives
    useEffect(() => { setEditState({}); setSimResult(null); }, [data?.name, data?.data]);

    // Smooth animate toward target (outside Canvas, so use rAF not useFrame)
    useEffect(() => {
        let raf;
        const animate = () => {
            setAnimatedFactor(prev => {
                const diff = explodeFactor - prev;
                if (Math.abs(diff) < 0.002) return explodeFactor;
                return prev + diff * 0.12;
            });
            raf = requestAnimationFrame(animate);
        };
        raf = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(raf);
    }, [explodeFactor]);

    useEffect(() => {
        if (data) console.log("CadWindow Data:", data.format, "material:", data.material, "parts:", data.parts?.length || 1);
    }, [data]);

    useEffect(() => {
        if (thoughtsEndRef.current) thoughtsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [thoughts]);

    const parseSTL = (b64) => {
        const byteCharacters = atob(b64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) byteNumbers[i] = byteCharacters.charCodeAt(i);
        const byteArray = new Uint8Array(byteNumbers);
        const loader = new STLLoader();
        const geom = loader.parse(byteArray.buffer);
        geom.center();
        return geom;
    };

    const partList = useMemo(() => {
        if (!data || data.format !== 'stl') return null;
        const raw = data.parts && data.parts.length ? data.parts : [{ name: data.name || 'part', data: data.data }];
        if (!data.data && !(data.parts && data.parts.length)) return null;
        try {
            return raw.map((p) => ({ name: p.name, geometry: parseSTL(p.data), materialKey: p.material || data.material }));
        } catch (e) {
            console.error("Failed to decode/parse STL:", e);
            return null;
        }
    }, [data]);

    const material = useMemo(() => MATERIALS[data?.material] || MATERIALS.default, [data]);

    const dims = useMemo(() => {
        if (!partList) return null;
        const box = new THREE.Box3();
        partList.forEach(({ geometry }) => {
            geometry.computeBoundingBox();
            if (geometry.boundingBox) box.union(geometry.boundingBox);
        });
        return { size: new THREE.Vector3().subVectors(box.max, box.min), center: box.getCenter(new THREE.Vector3()) };
    }, [partList]);

    const baseSpacing = useMemo(() => {
        if (!dims) return 40;
        const maxDim = Math.max(dims.size.x, dims.size.y, dims.size.z);
        return maxDim * 0.9 + 18; // smart base: larger assembly -> larger travel
    }, [dims]);

    const rig = useMemo(() => computeRig(dims), [dims]);
    const mx = rig ? rig.maxDim : 20; // every light/floor/shadow value is relative to this

    const handleGenerate = () => {
        if (!prompt.trim()) return;
        setIsSending(true);
        if (socket) socket.emit('generate_cad', { prompt });
        setPrompt("");
        setTimeout(() => setIsSending(false), 2000);
    };
    const handleIterate = () => {
        if (!prompt.trim()) return;
        setIsSending(true);
        if (socket) socket.emit('iterate_cad', { prompt });
        setIsIterating(false);
        setPrompt("");
        setIsSending(false);
    };

    const handleSimulate = async () => {
        if (!data || !partList || partList.length === 0) return;
        setSimulating(true);
        setSimResult(null);
        try {
            // Use first part's STL for demo (enterprise: would loop all parts)
            const b64 = data.parts && data.parts[0] ? data.parts[0].data : data.data;
            const res = await fetch('http://127.0.0.1:8000/simulate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ stl_b64: b64, material: data.material || 'steel', force_n: 500 })
            }).then(r=>r.json());
            setSimResult(res);
        } catch (e) {
            setSimResult({ok:false, error: String(e).slice(0,80)});
        } finally {
            setSimulating(false);
        }
    };

    const isMulti = partList && partList.length > 1;
    const isExploded = explodeFactor > 0.02;

    return (
        <div className="w-full h-full relative group bg-gray-900 rounded-lg overflow-hidden border border-cyan-500/30">
            <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                <button onClick={onClose} className="bg-red-500/20 hover:bg-red-500/50 text-red-500 p-1 rounded">X</button>
            </div>

            {/* Top Toolbar */}
            <div className="absolute top-2 left-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2 flex-wrap max-w-[85%]">
                <button onClick={() => setIsIterating(true)} className="bg-cyan-500/20 hover:bg-cyan-500/50 text-cyan-400 text-xs px-2 py-1 rounded border border-cyan-500/30 backdrop-blur-sm">ITERATE</button>
                {data && data.format === 'stl' && (
                    <>
                        <button onClick={() => setHoloMode(false)} className={`text-xs px-2 py-1 rounded border backdrop-blur-sm ${!holoMode ? 'bg-cyan-500/40 text-white border-cyan-400' : 'bg-gray-800/60 text-gray-300 border-gray-600 hover:bg-gray-700'}`}>REALISTIC</button>
                        <button onClick={() => setHoloMode(true)} className={`text-xs px-2 py-1 rounded border backdrop-blur-sm ${holoMode ? 'bg-cyan-400/40 text-cyan-100 border-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.5)]' : 'bg-gray-800/60 text-gray-300 border-gray-600 hover:bg-gray-700'}`}>HOLOGRAM</button>
                        <button onClick={() => setShowDims(v => !v)} className={`text-xs px-2 py-1 rounded border backdrop-blur-sm ${showDims ? 'bg-cyan-500/30 text-cyan-200 border-cyan-400/50' : 'bg-gray-800/60 text-gray-400 border-gray-600 hover:bg-gray-700'}`}>DIMS {showDims ? 'ON' : 'OFF'}</button>
                        {data.plan && (
                            <button onClick={() => setShowTeam(v => !v)} className={`text-xs px-2 py-1 rounded border backdrop-blur-sm ${showTeam ? 'bg-violet-400/40 text-violet-100 border-violet-300' : 'bg-gray-800/60 text-gray-300 border-gray-600 hover:bg-gray-700'}`}>
                                PLAN{(data.findings || []).filter(f => !f.ok).length ? ` · ${(data.findings || []).filter(f => !f.ok).length}!` : ''}
                            </button>
                        )}
                        {isMulti && (
                            <>
                                <button onClick={() => setExplodeFactor(v => v > 0.5 ? 0 : 1)} className={`text-xs px-2 py-1 rounded border backdrop-blur-sm ${isExploded ? 'bg-amber-400/40 text-amber-100 border-amber-300 shadow-[0_0_10px_rgba(251,146,60,0.4)]' : 'bg-gray-800/60 text-gray-300 border-gray-600 hover:bg-gray-700'}`}>{isExploded ? 'ASSEMBLE' : 'EXPLODE'}</button>
                                <button onClick={() => setShowTrails(v=>!v)} className={`text-[10px] px-2 py-1 rounded border backdrop-blur-sm ${showTrails ? 'bg-cyan-500/30 text-cyan-200 border-cyan-400/50' : 'bg-gray-800/40 text-gray-400 border-gray-700'}`}>TRAILS {showTrails ? 'ON' : 'OFF'}</button>
                                <button onClick={() => setShowGhost(v=>!v)} className={`text-[10px] px-2 py-1 rounded border backdrop-blur-sm ${showGhost ? 'bg-cyan-500/30 text-cyan-200 border-cyan-400/50' : 'bg-gray-800/40 text-gray-400 border-gray-700'}`}>GHOST {showGhost ? 'ON' : 'OFF'}</button>
                            </>
                        )}
                        <button onClick={handleSimulate} disabled={simulating} className={`text-[10px] px-2 py-1 rounded border backdrop-blur-sm font-mono ${simulating ? 'bg-red-500/30 text-red-200 border-red-400/50 animate-pulse' : simResult ? (simResult.status==='PASS' ? 'bg-green-500/30 text-green-200 border-green-400/50' : simResult.status==='FAIL' ? 'bg-red-500/30 text-red-200 border-red-400/50' : 'bg-yellow-500/30 text-yellow-200 border-yellow-400/50') : 'bg-violet-500/20 text-violet-300 border-violet-400/30 hover:bg-violet-500/30'}`}>{simulating ? 'SIMULATING...' : simResult ? `SIM ${simResult.status} SF${simResult.safety_factor}` : 'SIMULATE 500N'}</button>
                    </>
                )}
            </div>

            {/* Explode Slider — only for assemblies */}
            {isMulti && (
                <div className="absolute bottom-14 left-3 right-3 z-10 flex items-center gap-2 bg-black/55 backdrop-blur-md border border-cyan-500/20 rounded-full px-3 py-2">
                    <span className="text-[9px] font-mono text-cyan-300 tracking-widest whitespace-nowrap">ASSEMBLED</span>
                    <input type="range" min={0} max={100} value={Math.round(explodeFactor*100)} onChange={e=>setExplodeFactor(parseInt(e.target.value)/100)} className="flex-1 accent-cyan-400 h-1" />
                    <span className="text-[9px] font-mono text-amber-300 tracking-widest whitespace-nowrap">EXPLODED</span>
                    <span className="text-[10px] font-mono text-white/70 w-8 text-right">{Math.round(explodeFactor*100)}%</span>
                </div>
            )}
            {/* Level 3 plan panel — what was decided before the build, and what the
                inspector found afterwards. Open findings are shown first, because a
                build that reports "passed" with three envelopes in it is the failure
                this panel exists to make impossible. */}
            {showTeam && data?.plan && (
                <div className="absolute top-12 left-2 z-20 w-[300px] max-h-[70%] overflow-y-auto bg-black/80 backdrop-blur-md border border-violet-500/30 rounded-lg p-3 text-[10px] font-mono text-violet-100 space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="font-bold tracking-widest text-violet-300">ENGINEERING PLAN</span>
                        <span className="opacity-60">{data.plan.source}</span>
                    </div>
                    <p className="text-white/80 leading-snug">{data.plan.function}</p>
                    {data.plan.envelope_mm?.length === 3 && (
                        <div>ENVELOPE · {data.plan.envelope_mm.map(v => Math.round(v)).join(' × ')} mm</div>
                    )}
                    <div>PARTS · {data.plan.parts.length}{data.clarifications?.length ? ` · clarification×${data.clarifications.length}` : ''}</div>
                    {data.plan.interfaces?.length > 0 && (
                        <div className="text-amber-200/90">INTERFACES · {data.plan.interfaces.join(' · ')}</div>
                    )}
                    {data.plan.checks?.length > 0 && (
                        <div className="text-cyan-200/90">CHECKS · {data.plan.checks.join(' · ')}</div>
                    )}
                    {(data.findings || []).length > 0 && (
                        <div className="border-t border-violet-500/20 pt-2 space-y-1">
                            <div className="font-bold tracking-widest text-violet-300">INSPECTOR</div>
                            {data.findings.filter(f => !f.ok).map((f, i) => (
                                <div key={i} className="text-red-300">✗ {f.subject}: {f.detail}</div>
                            ))}
                            {data.findings.filter(f => f.ok).map((f, i) => (
                                <div key={`ok${i}`} className="text-emerald-300/80">✓ {f.subject}: {f.detail}</div>
                            ))}
                        </div>
                    )}
                    {(data.refinements || []).length > 0 && (
                        <div className="border-t border-violet-500/20 pt-2 space-y-1">
                            <div className="font-bold tracking-widest text-violet-300">REFINER</div>
                            {data.refinements.map((r, i) => (
                                <div key={i} className={r.improved ? 'text-emerald-300' : 'text-gray-400'}>
                                    {r.improved ? '↑' : '·'} {r.action}: {r.before} → {r.after}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Engineering Evidence Bar — measured facts, not adjectives */}
            {data?.evidence && data.format === 'stl' && (
                <div className={`absolute bottom-3 left-3 right-3 z-10 flex items-center justify-between px-3 py-1.5 rounded-full text-[10px] font-mono backdrop-blur-md border ${simResult ? 'bottom-12' : ''} ${data.evidence.status === 'PASS' ? 'bg-emerald-500/15 border-emerald-400/40 text-emerald-200' : data.evidence.status === 'FAIL' ? 'bg-red-500/15 border-red-400/40 text-red-200' : 'bg-cyan-500/10 border-cyan-400/30 text-cyan-200'}`}>
                    <span className="font-bold">{data.evidence.status} · {data.evidence.parts}_PARTS</span>
                    <span className="opacity-80 truncate mx-2">{(data.evidence.verdicts || []).map(v => `${v.test}:${v.status}`).join(' · ') || data.evidence.note || 'measured at build'}</span>
                    <span className="text-[9px] opacity-70 whitespace-nowrap">ENGINEERING_EVIDENCE</span>
                </div>
            )}
            {/* Simulation Result Bar — real FEA */}
            {simResult && (
                <div className={`absolute bottom-3 left-3 right-3 z-10 flex items-center justify-between px-3 py-1.5 rounded-full text-[10px] font-mono backdrop-blur-md border ${simResult.status==='PASS' ? 'bg-green-500/15 border-green-400/40 text-green-200' : simResult.status==='FAIL' ? 'bg-red-500/15 border-red-400/40 text-red-200' : simResult.status==='WARN' ? 'bg-yellow-500/15 border-yellow-400/40 text-yellow-200' : 'bg-gray-700/50 border-gray-600 text-gray-300'}`}>
                    <span>{simResult.material || 'steel'} · {simResult.max_stress_mpa} MPa / {simResult.yield_strength_mpa} MPa yield</span>
                    <span className="font-bold">SF {simResult.safety_factor} · {simResult.status}</span>
                    <span className="text-[9px] opacity-70">500N load</span>
                </div>
            )}

            {/* Generation Overlay */}
            {(isIterating || (!data && data?.format !== 'loading')) && (
                <div className={`absolute inset-0 z-20 ${!data ? 'bg-gray-900' : 'bg-black/80'} flex items-center justify-center p-4`}>
                    <div className="bg-gray-800 border border-cyan-500/50 rounded p-4 w-full max-w-sm pointer-events-auto shadow-[0_0_20px_rgba(6,182,212,0.2)]">
                        <h4 className="text-cyan-400 text-sm mb-2 font-mono">{!data ? "New Design" : "Refine Design"}</h4>
                        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={!data ? "Describe what you want to create..." : "e.g., Make the wheels bigger..."} className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white text-sm mb-3 focus:outline-none focus:border-cyan-500 h-24 resize-none" autoFocus onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); !data ? handleGenerate() : handleIterate(); }}} />
                        <div className="flex justify-end gap-2">
                            {data && <button onClick={() => setIsIterating(false)} className="text-gray-400 text-xs hover:text-white px-2 py-1">Cancel</button>}
                            <button onClick={!data ? handleGenerate : handleIterate} disabled={isSending} className="bg-cyan-600 hover:bg-cyan-500 text-white text-xs px-3 py-1 rounded">{isSending ? "Generating..." : (!data ? "Generate" : "Update")}</button>
                        </div>
                    </div>
                </div>
            )}

            <Canvas
                shadows
                dpr={[1, 2]}
                gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.05 }}
                camera={{ position: [7, 5.5, 9], fov: 38 }}
            >
                <color attach="background" args={['#07090d']} />
                {rig && <fog attach="fog" args={['#07090d', rig.dist * 1.6, rig.dist * 4.2]} />}
                <CameraRig rig={rig} />

                {/* Stark-lab studio. The old `<Environment preset="studio" />` fetched its
                    HDR from a CDN, so reflections vanished with the network down and the look
                    changed with drei's asset host. Lightformers bake the same rig locally. */}
                <Suspense fallback={null}>
                    <Environment key={holoMode ? 'holo' : 'studio'} resolution={256} frames={1} background={false} environmentIntensity={holoMode ? 0.45 : 1.05}>
                        {holoMode ? (
                            <>
                                <Lightformer form="ring" intensity={0.9} color="#0ea5e9" scale={rig ? rig.radius * 1.6 : 8} position={[0, 6, -6]} />
                                <Lightformer form="rect" intensity={0.4} color="#22d3ee" scale={[12, 4, 1]} position={[-6, 2, -4]} rotation-y={Math.PI / 3} />
                            </>
                        ) : (
                            <>
                                <Lightformer form="rect" intensity={3.0} color="#ffffff" scale={[10, 6, 1]} position={[0, 6, -8]} />
                                <Lightformer form="rect" intensity={1.3} color="#bcd7ff" scale={[10, 4, 1]} position={[-6, 1, 4]} rotation-y={Math.PI / 2} />
                                <Lightformer form="rect" intensity={1.1} color="#ffe6c2" scale={[8, 4, 1]} position={[7, 2, 2]} rotation-y={-Math.PI / 2} />
                                <Lightformer form="rect" intensity={0.5} color="#ffffff" scale={[12, 12, 1]} position={[0, -4, 0]} rotation-x={Math.PI / 2} />
                                <Lightformer form="ring" intensity={1.6} color="#ffffff" scale={5} position={[0, 8, 0]} />
                            </>
                        )}
                    </Environment>
                </Suspense>
                <directionalLight
                    position={[mx * 0.9, mx * 1.7, mx * 0.6]}
                    intensity={holoMode ? 0.5 : 2.2}
                    castShadow
                    shadow-mapSize={[2048, 2048]}
                    shadow-bias={-0.0004}
                    shadow-camera-near={mx * 0.1}
                    shadow-camera-far={mx * 6}
                    shadow-camera-left={-mx}
                    shadow-camera-right={mx}
                    shadow-camera-top={mx}
                    shadow-camera-bottom={-mx}
                />
                <directionalLight position={[-mx * 1.1, mx * 0.6, -mx * 0.9]} intensity={holoMode ? 0.25 : 0.85} color="#8ecbff" />
                <spotLight position={[0, mx * 1.9, 0]} angle={0.5} penumbra={1} intensity={holoMode ? 0.3 : 1.2} color="#fff4e0" />

                {data?.format === 'loading' ? <LoadingCube /> : (
                    partList && (
                        <>
                        <Center>
                                {partList.map((part, i) => {
                                    const asmPart = data.assembly && data.assembly.parts && data.assembly.parts[i] ? data.assembly.parts[i] : null;
                                    const a = asmPart ? asmPart.axis : null;
                                    const es = editState[part.name];
                                    if (es && es.visible === false) return null; // surgery: removed/hidden
                                    const eff = es && es.color ? { ...MATERIALS[data.material] || MATERIALS.default, color: es.color, metalness: 0.6, roughness: 0.3 } : null;
                                    const scale = es?.scale && es.scale !== 1 ? es.scale : 1;
                                    // Floating part callout for hologram mode: index, material, real size
                                    const sizeLabel = asmPart?.size ? ` · ${asmPart.size.map(v => Number(v).toFixed(0)).join('×')} mm` : '';
                                    const holoLabel = `P${i + 1} · ${(part.materialKey || 'metal').toUpperCase()}${sizeLabel}`;
                                    return (
                                        <group key={part.name} scale={scale}
                                            onPointerDown={(e) => { pickDownRef.current = { x: e.clientX, y: e.clientY }; }}
                                        >
                                            <ExplodedPart part={part} index={i} total={partList.length} material={eff || material} hologram={holoMode} dims={dims} explodeFactor={explodeFactor} animatedFactor={animatedFactor} showTrails={showTrails && isExploded} showGhost={showGhost} spacing={baseSpacing} assemblyAxis={a} simResult={simResult} selected={selectedPart === part.name} onPick={handlePartPick} label={holoLabel} />
                                        </group>
                                    );
                                })}
                        </Center>
                        {showDims && !isExploded && rig && dims && <DimensionOverlay dims={dims} rig={rig} />}
                        {holoMode && rig && <ScannerRing rig={rig} />}
                        {holoMode && rig && dims && <HoloReadout name={data?.name} dims={dims} rig={rig} material={data?.material} />}
                        </>
                    )
                )}

                {/* Grounding: soft contact shadow + Stark-lab reflective floor */}
                <ContactShadows position={[0, (rig ? rig.groundY : 0) + 0.001, 0]} opacity={holoMode ? 0.3 : 0.62} scale={mx * 3} blur={2.6} far={mx * 1.2} resolution={1024} color="#000000" />
                <mesh rotation-x={-Math.PI / 2} position={[0, (rig ? rig.groundY : 0) - 0.01, 0]} receiveShadow>
                    <planeGeometry args={[mx * 60, mx * 60]} />
                    <MeshReflectorMaterial
                        blur={[320, 90]}
                        resolution={1024}
                        mixBlur={1}
                        mixStrength={holoMode ? 4 : 14}
                        roughness={0.86}
                        depthScale={1.15}
                        minDepthThreshold={0.4}
                        maxDepthThreshold={1.35}
                        color="#0a0d12"
                        metalness={0.55}
                        mirror={holoMode ? 0.25 : 0.65}
                    />
                </mesh>

                {holoMode && dims && <Sparkles count={60} scale={[dims.size.x * 1.8, dims.size.y * 1.8, dims.size.z * 1.8]} size={2.2} speed={0.4} opacity={0.55} color="#22d3ee" />}
                <OrbitControls autoRotate={!isIterating && !isExploded} autoRotateSpeed={1} makeDefault minPolarAngle={0.2} maxPolarAngle={Math.PI / 2.05} minDistance={mx * 0.5} maxDistance={mx * 25} />
            </Canvas>

            {data?.format === 'loading' && (
                <div className="absolute inset-y-0 right-0 w-2/5 p-4 bg-black/70 backdrop-blur-sm border-l border-green-500/30 overflow-hidden flex flex-col">
                    <div className="flex items-center justify-between mb-2">
                        <h4 className="text-green-400 text-xs font-mono tracking-widest uppercase flex items-center gap-2"><span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>Designer Thinking...</h4>
                        {retryInfo.attempt && <span className={`text-xs font-mono px-2 py-0.5 rounded ${retryInfo.error ? 'bg-yellow-500/20 text-yellow-400' : 'bg-cyan-500/20 text-cyan-400'}`}>Attempt {retryInfo.attempt}/{retryInfo.maxAttempts || 3}</span>}
                    </div>
                    {retryInfo.error && <div className="mb-2 p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs font-mono"><span className="text-red-500 font-bold">⚠ Error:</span> {retryInfo.error}</div>}
                    <div className="flex-1 overflow-y-auto text-green-400/80 text-xs font-mono whitespace-pre-wrap leading-relaxed scrollbar-thin scrollbar-thumb-green-500/30">{thoughts}<div ref={thoughtsEndRef} /></div>
                </div>
            )}

            {data && data.format === 'stl' && dims && (
                <div className="absolute bottom-2 left-2 z-10 pointer-events-none space-y-1">
                    <div className="text-[11px] text-cyan-300/90 font-mono tracking-widest">{holoMode ? 'HOLOGRAPHIC_VISUAL' : (data.name ? data.name.toUpperCase() : 'DESIGN')}</div>
                    <div className="text-[10px] font-mono tracking-widest">{selectedPart ? <span className="text-amber-300 animate-pulse">◎ TARGET: {selectedPart.toUpperCase()} — say "remove this"</span> : partList && partList.length > 1 ? `${partList.length}_PART_ASSEMBLY${isExploded ? ` · EXPLODED ${Math.round(explodeFactor*100)}%` : ''}` : holoMode ? 'PROJECTION_ACTIVE' : `${(data.material || 'default').toUpperCase()}_MATERIAL`}</div>
                    <div className="text-[10px] text-gray-400/80 font-mono">{dims.size.x.toFixed(1)} × {dims.size.y.toFixed(1)} × {dims.size.z.toFixed(1)} mm</div>
                </div>
            )}
            <div className="absolute bottom-2 right-2 text-[10px] text-cyan-500/50 font-mono tracking-widest pointer-events-none">CAD_ENGINE_V2: {data?.format?.toUpperCase() || "READY"}</div>
        </div>
    );
};

export default CadWindow;
