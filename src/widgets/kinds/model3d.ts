/**
 * A small 3D viewer: wireframe objects that turn slowly, drawn on a canvas of
 * its own. Objects come as data: box, cylinder, sphere, line.
 */
import * as THREE from "three";
import type { KindSpec, Props, WidgetEnv, WidgetInstance } from "../types";
import { arr, h, num, obj, str } from "../dom";
import { resolvePalette } from "../palette";

const SIZE = 400;
const MAX_OBJECTS = 80;

interface Part { mesh: THREE.Group; home: THREE.Vector3; own: string | null; mats: THREE.Material[] }

function vec(v: unknown, def: [number, number, number]): THREE.Vector3 {
  const a = arr(v, 3);
  return new THREE.Vector3(num(a[0], def[0], -1e4, 1e4), num(a[1], def[1], -1e4, 1e4), num(a[2], def[2], -1e4, 1e4));
}

function makeModel(props: Props, env: WidgetEnv): WidgetInstance {
  const el = h("div", "wd");
  el.style.filter = "drop-shadow(0 0 6px rgba(var(--rgb), 0.45))";
  const canvas = document.createElement("canvas");
  el.appendChild(canvas);

  let renderer: THREE.WebGLRenderer | null = null;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  } catch {
    el.appendChild(h("div", "wd-t wd-cap", "3D UNAVAILABLE")).style.top = "190px";
  }
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (renderer) { renderer.setPixelRatio(dpr); renderer.setSize(SIZE, SIZE, true); renderer.setClearColor(0x000000, 0); }
  canvas.style.width = `${SIZE}px`; canvas.style.height = `${SIZE}px`; canvas.style.display = "block";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 5000);
  const world = new THREE.Group();
  scene.add(world);

  // A faint floor ring so the object has something to stand on.
  const floorMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 });
  const floor = new THREE.Group();
  scene.add(floor);

  let parts: Part[] = [];
  let radius = 1;
  let spin = 0.12, explode = 0, explodeShown = 0;
  let angle = 0.6;
  const mats = new Set<THREE.Material>();
  const geos: THREE.BufferGeometry[] = [];

  const clear = () => {
    for (const p of parts) world.remove(p.mesh);
    parts = [];
    for (const g of geos) g.dispose();
    geos.length = 0;
    for (const m of mats) m.dispose();
    mats.clear();
  };

  const colourFor = (own: string | null): THREE.Color => new THREE.Color(own ?? env.pal.c);

  const build = (p: Props) => {
    clear();
    spin = num(p.spin, 0.12, -2, 2);
    explode = num(p.explode, 0, 0, 1);
    const box = new THREE.Box3();
    arr(p.objects, MAX_OBJECTS).forEach((raw) => {
      const o = obj(raw);
      const kind = str(o.kind, "box", 10);
      const pos = vec(o.pos, [0, 0, 0]);
      const rot = vec(o.rot, [0, 0, 0]);
      const own = o.color !== undefined && str(o.color, "") !== "" ? resolvePalette(o.color).c : null;
      let geo: THREE.BufferGeometry;
      const sizeArr = arr(o.size, 3);
      const sz = Array.isArray(o.size)
        ? new THREE.Vector3(num(sizeArr[0], 10, 0.01, 1e4), num(sizeArr[1], num(sizeArr[0], 10), 0.01, 1e4), num(sizeArr[2], num(sizeArr[0], 10), 0.01, 1e4))
        : new THREE.Vector3().setScalar(num(o.size, 10, 0.01, 1e4));
      const r = num(o.r, 5, 0.01, 1e4);
      const group = new THREE.Group();
      if (kind === "line") {
        const to = Array.isArray(o.to) ? vec(o.to, [0, 0, 0]) : pos.clone().add(sz);
        geo = new THREE.BufferGeometry().setFromPoints([pos, to]);
        const m = new THREE.LineBasicMaterial({ color: colourFor(own), transparent: true, opacity: 0.95 });
        group.add(new THREE.Line(geo, m));
        mats.add(m); geos.push(geo);
        box.expandByPoint(pos); box.expandByPoint(to);
        parts.push({ mesh: group, home: new THREE.Vector3(), own, mats: [m] });
        world.add(group);
        return;
      }
      if (kind === "sphere") geo = new THREE.SphereGeometry(r, 18, 12);
      else if (kind === "cylinder") geo = new THREE.CylinderGeometry(r, r, sz.y, 24, 1);
      else geo = new THREE.BoxGeometry(sz.x, sz.y, sz.z);
      const edges = new THREE.EdgesGeometry(geo, kind === "sphere" ? 1 : 25);
      const lineM = new THREE.LineBasicMaterial({ color: colourFor(own), transparent: true, opacity: 0.95 });
      const fillM = new THREE.MeshBasicMaterial({ color: colourFor(own), transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide });
      group.add(new THREE.LineSegments(edges, lineM), new THREE.Mesh(geo, fillM));
      group.position.copy(pos);
      group.rotation.set(THREE.MathUtils.degToRad(rot.x), THREE.MathUtils.degToRad(rot.y), THREE.MathUtils.degToRad(rot.z));
      geos.push(geo, edges); mats.add(lineM); mats.add(fillM);
      box.expandByObject(group);
      parts.push({ mesh: group, home: pos.clone(), own, mats: [lineM, fillM] });
      world.add(group);
    });
    if (parts.length === 0) { box.set(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(5, 5, 5)); }
    const centre = box.getCenter(new THREE.Vector3());
    radius = Math.max(1e-3, box.getSize(new THREE.Vector3()).length() / 2);
    for (const p of parts) { p.home.sub(centre); p.mesh.position.copy(p.home); }
    // Lines carry world coordinates: shift their group instead.
    for (const p of parts) if (p.home.lengthSq() === 0) p.mesh.position.copy(centre.clone().negate());
    camera.position.set(0, radius * 0.9, radius * 3.6);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    // Floor rings.
    floor.clear();
    for (const k of [1.2, 1.6]) {
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 64; i++) pts.push(new THREE.Vector3(Math.cos((i / 64) * Math.PI * 2) * radius * k, -radius * 0.95, Math.sin((i / 64) * Math.PI * 2) * radius * k));
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      geos.push(g);
      floor.add(new THREE.Line(g, floorMat));
    }
  };
  build(props);

  const title = h("div", "wd-t wd-cap", str(props.title, "", 40).toUpperCase());
  title.style.top = "372px";
  el.appendChild(title);

  return {
    el,
    update(p) {
      if (p.objects !== undefined) build({ ...p, spin: p.spin ?? spin });
      else { if (p.spin !== undefined) spin = num(p.spin, spin, -2, 2); if (p.explode !== undefined) explode = num(p.explode, 0, 0, 1); }
      title.textContent = str(p.title, title.textContent ?? "", 40).toUpperCase();
    },
    tick(_t, dt) {
      if (!renderer) return;
      angle += spin * Math.PI * 2 * dt;
      world.rotation.y = angle;
      floor.rotation.y = -angle * 0.4;
      explodeShown += (explode - explodeShown) * (1 - Math.exp(-4 * dt));
      for (const p of parts) {
        if (p.home.lengthSq() > 0) p.mesh.position.copy(p.home).multiplyScalar(1 + explodeShown * 1.2);
        for (const m of p.mats) if (!p.own && "color" in m) (m as THREE.LineBasicMaterial).color.set(env.pal.c);
      }
      floorMat.color.set(env.pal.cd);
      renderer.render(scene, camera);
    },
    busy: () => false,
    destroy() {
      clear();
      floor.clear();
      floorMat.dispose();
      if (renderer) { renderer.dispose(); renderer.forceContextLoss(); }
    },
  };
}

export const MODEL: Record<string, KindSpec> = {
  model3d: { design: [SIZE, SIZE], inner: 1.6, around: [1.15, 1.15], panel: false, make: makeModel },
};
