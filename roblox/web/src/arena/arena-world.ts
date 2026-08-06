import {
  Color3,
  MeshBuilder,
  type Scene,
  StandardMaterial,
  TransformNode,
  Vector3,
} from "@babylonjs/core";
import { cssArenaColor } from "./arena-palette";
import type { ArenaPositionId } from "./encounter";
import type {
  ArenaAttackPresentation,
  ArenaPositionPresentation,
} from "./presentation";

const color = {
  void: Color3.FromHexString(cssArenaColor("void")),
  stage: Color3.FromHexString(cssArenaColor("stage")),
  panel: Color3.FromHexString(cssArenaColor("panel")),
  cyan: Color3.FromHexString(cssArenaColor("cyan")),
  gold: Color3.FromHexString(cssArenaColor("gold")),
  violet: Color3.FromHexString(cssArenaColor("violet")),
  danger: Color3.FromHexString(cssArenaColor("danger")),
} as const;

const material = (
  scene: Scene,
  name: string,
  diffuse: Color3,
  emissive = Color3.Black(),
): StandardMaterial => {
  const result = new StandardMaterial(name, scene);
  result.diffuseColor = diffuse;
  result.emissiveColor = emissive;
  result.specularColor = color.cyan.scale(0.08);
  return result;
};

export const anchorPosition: Record<ArenaPositionId, Vector3> = {
  shelter: new Vector3(-0.78, 0, -3.55),
  midline: new Vector3(0, 0, -3.15),
  spotlight: new Vector3(0.78, 0, -2.75),
};

type AnchorVisual = {
  readonly root: TransformNode;
  readonly rim: StandardMaterial;
  readonly core: StandardMaterial;
};

export type ArenaWorld = {
  readonly player: PerformerRig;
  readonly anchors: Readonly<Record<ArenaPositionId, AnchorVisual>>;
  setPositionStates(states: readonly ArenaPositionPresentation[]): void;
  setAttack(attack: ArenaAttackPresentation | undefined): void;
};

export type PerformerRig = {
  readonly root: TransformNode;
  readonly leftArm: TransformNode;
  readonly rightArm: TransformNode;
  readonly focus: TransformNode;
  readonly ward: TransformNode;
};

const createAnchor = (scene: Scene, id: ArenaPositionId): AnchorVisual => {
  const root = new TransformNode(`anchor-${id}`, scene);
  root.position.copyFrom(anchorPosition[id]);
  const rim = material(scene, `${id}-rim`, color.panel, color.cyan.scale(0.12));
  const core = material(scene, `${id}-core`, color.stage);
  const disc = MeshBuilder.CreateCylinder(
    `${id}-disc`,
    { diameter: 1.45, height: 0.14, tessellation: 10 },
    scene,
  );
  disc.parent = root;
  disc.position.y = 0.08;
  disc.material = core;
  const ring = MeshBuilder.CreateTorus(
    `${id}-ring`,
    { diameter: 1.35, thickness: 0.07, tessellation: 32 },
    scene,
  );
  ring.parent = root;
  ring.position.y = 0.17;
  ring.material = rim;
  if (id === "shelter") {
    for (const x of [-0.82, 0.82]) {
      const pillar = MeshBuilder.CreateBox(
        "shelter-pillar",
        { width: 0.26, height: 1.15, depth: 0.36 },
        scene,
      );
      pillar.parent = root;
      pillar.position = new Vector3(x * 0.7, 0.62, 0.18);
      pillar.rotation.z = x * -0.14;
      pillar.material = core;
    }
    const lintel = MeshBuilder.CreateBox(
      "shelter-lintel",
      { width: 1.25, height: 0.22, depth: 0.4 },
      scene,
    );
    lintel.parent = root;
    lintel.position = new Vector3(0, 1.18, 0.18);
    lintel.material = core;
  } else if (id === "midline") {
    for (const x of [-0.52, 0.52]) {
      const stone = MeshBuilder.CreateBox(
        "midline-stone",
        { width: 0.22, height: 0.72, depth: 0.34 },
        scene,
      );
      stone.parent = root;
      stone.position = new Vector3(x * 0.82, 0.4, 0.2);
      stone.rotation.z = x * 0.18;
      stone.material = core;
    }
  } else {
    const beam = MeshBuilder.CreateCylinder(
      "spotlight-beam",
      { diameterTop: 0.16, diameterBottom: 1.1, height: 1.8, tessellation: 16 },
      scene,
    );
    beam.parent = root;
    beam.position.y = 0.95;
    beam.material = rim;
    beam.visibility = 0.16;
  }
  return { root, rim, core };
};

const createPerformer = (scene: Scene): PerformerRig => {
  const root = new TransformNode("rift-performer", scene);
  root.position.copyFrom(anchorPosition.midline);
  root.scaling.setAll(0.6);
  const suit = material(
    scene,
    "performer-suit",
    color.panel,
    color.cyan.scale(0.1),
  );
  const energy = material(
    scene,
    "performer-energy",
    color.cyan.scale(0.25),
    color.cyan,
  );
  const skin = material(scene, "performer-skin", color.gold.scale(0.7));
  const torso = MeshBuilder.CreateCylinder(
    "performer-torso",
    { diameterTop: 0.48, diameterBottom: 0.68, height: 1.05, tessellation: 8 },
    scene,
  );
  torso.parent = root;
  torso.position.y = 1.2;
  torso.material = suit;
  const head = MeshBuilder.CreateSphere(
    "performer-head",
    { diameter: 0.5, segments: 10 },
    scene,
  );
  head.parent = root;
  head.position.y = 2.05;
  head.material = skin;
  const crest = MeshBuilder.CreatePolyhedron(
    "performer-crest",
    { type: 1, size: 0.32 },
    scene,
  );
  crest.parent = root;
  crest.position = new Vector3(0, 2.38, 0);
  crest.scaling = new Vector3(0.7, 1.4, 0.7);
  crest.material = energy;
  for (const x of [-0.24, 0.24]) {
    const leg = MeshBuilder.CreateCylinder(
      "performer-leg",
      { diameter: 0.2, height: 0.9, tessellation: 8 },
      scene,
    );
    leg.parent = root;
    leg.position = new Vector3(x, 0.48, 0);
    leg.material = suit;
  }
  const arm = (name: string, x: number): TransformNode => {
    const pivot = new TransformNode(name, scene);
    pivot.parent = root;
    pivot.position = new Vector3(x, 1.58, 0);
    const mesh = MeshBuilder.CreateCylinder(
      `${name}-mesh`,
      { diameter: 0.18, height: 0.82, tessellation: 8 },
      scene,
    );
    mesh.parent = pivot;
    mesh.position.y = -0.34;
    mesh.material = suit;
    return pivot;
  };
  const leftArm = arm("performer-left-arm", -0.46);
  const rightArm = arm("performer-right-arm", 0.46);
  const focus = new TransformNode("performance-focus", scene);
  focus.parent = root;
  focus.position = new Vector3(0, 1.22, -0.46);
  const focusRing = MeshBuilder.CreateTorus(
    "focus-ring",
    { diameter: 1, thickness: 0.1, tessellation: 24 },
    scene,
  );
  focusRing.parent = focus;
  focusRing.rotation.x = Math.PI / 2;
  focusRing.material = energy;
  const ward = new TransformNode("performer-ward", scene);
  ward.parent = root;
  const wardMesh = MeshBuilder.CreateSphere(
    "ward-shell",
    { diameter: 2.2, segments: 16, slice: 0.55 },
    scene,
  );
  wardMesh.parent = ward;
  wardMesh.position.y = 1.08;
  wardMesh.material = energy;
  wardMesh.visibility = 0;
  return { root, leftArm, rightArm, focus, ward };
};

export const buildArenaWorld = (scene: Scene): ArenaWorld => {
  const floorMaterial = material(
    scene,
    "arena-floor",
    color.void,
    color.violet.scale(0.035),
  );
  const floor = MeshBuilder.CreateCylinder(
    "arena-stage",
    { diameter: 13, height: 0.22, tessellation: 48 },
    scene,
  );
  floor.position.y = -0.16;
  floor.material = floorMaterial;
  const anchors = {
    shelter: createAnchor(scene, "shelter"),
    midline: createAnchor(scene, "midline"),
    spotlight: createAnchor(scene, "spotlight"),
  };
  const player = createPerformer(scene);
  const telegraphMaterial = material(
    scene,
    "attack-telegraph",
    color.danger.scale(0.2),
    color.danger,
  );
  telegraphMaterial.alpha = 0.5;
  const sweepPath = MeshBuilder.CreateBox(
    "rift-sweep-path",
    { width: 5.2, height: 0.05, depth: 0.58 },
    scene,
  );
  sweepPath.position = new Vector3(0.15, 0.2, -3.1);
  sweepPath.material = telegraphMaterial;
  sweepPath.visibility = 0;
  const burstTarget = MeshBuilder.CreateTorus(
    "void-burst-target",
    { diameter: 4.2, thickness: 0.18, tessellation: 48 },
    scene,
  );
  burstTarget.position = new Vector3(-0.8, 0.23, -3.3);
  burstTarget.material = telegraphMaterial;
  burstTarget.visibility = 0;
  return {
    player,
    anchors,
    setPositionStates: (states) => {
      for (const { current, id, state } of states) {
        const visual = anchors[id];
        const energy =
          state === "danger"
            ? color.danger
            : state === "safe"
              ? color.cyan
              : color.violet;
        visual.rim.emissiveColor = energy.scale(
          state === "neutral" ? 0.12 : 0.8,
        );
        visual.rim.diffuseColor = energy.scale(0.28);
        visual.root.scaling.setAll(current ? 1.08 : 1);
      }
    },
    setAttack: (attack) => {
      sweepPath.visibility = 0;
      burstTarget.visibility = 0;
      if (!attack) return;
      const visibility =
        attack.phase === "impact"
          ? 0.92
          : attack.phase === "recovery"
            ? Math.max(0, 0.42 * (1 - attack.progress))
            : 0.22 + attack.progress * 0.48;
      const target = attack.type === "sweep" ? sweepPath : burstTarget;
      target.visibility = visibility;
      if (attack.type === "sweep") {
        sweepPath.scaling.z = 0.35 + attack.progress * 0.65;
      } else {
        const scale = 0.55 + attack.progress * 0.45;
        burstTarget.scaling.setAll(scale);
      }
    },
  };
};
