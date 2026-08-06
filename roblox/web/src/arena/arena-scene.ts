import "@babylonjs/loaders/glTF";
import {
  type AnimationGroup,
  Color3,
  Color4,
  DirectionalLight,
  Engine,
  HemisphericLight,
  Scene,
  SceneLoader,
  UniversalCamera,
  Vector3,
} from "@babylonjs/core";
import { cssArenaColor } from "./arena-palette";
import {
  type ArenaWorld,
  anchorPosition,
  buildArenaWorld,
} from "./arena-world";
import type { ArenaEffect, ArenaRunState } from "./combat";
import type { ArenaEncounter } from "./encounter";
import { deriveArenaPresentation } from "./presentation";

type BossSemanticClip =
  | "intro"
  | "idle"
  | "sweep-prepare"
  | "sweep-impact"
  | "burst-prepare"
  | "burst-impact"
  | "hit"
  | "opening"
  | "phase"
  | "defeat";

const bossClips: Record<BossSemanticClip, string> = {
  intro: "Wave",
  idle: "Idle",
  "sweep-prepare": "No",
  "sweep-impact": "Weapon",
  "burst-prepare": "Jump_Idle",
  "burst-impact": "Punch",
  hit: "HitReact",
  opening: "Duck",
  phase: "Yes",
  defeat: "Death",
};

export class ArenaScene {
  private readonly engine: Engine;
  private readonly scene: Scene;
  private readonly camera: UniversalCamera;
  private readonly world: ArenaWorld;
  private readonly reducedMotion: boolean;
  private readonly bossAnimations = new Map<string, AnimationGroup>();
  private currentBossClip = "";
  private actionUntil = 0;
  private action: "perform" | "hit" | "idle" = "idle";
  private cameraImpulseUntil = 0;
  private loaded = false;
  private disposed = false;

  constructor(canvas: HTMLCanvasElement) {
    const engine = new Engine(canvas, true, {
      stencil: true,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    let scene: Scene | undefined;
    try {
      scene = new Scene(engine);
      scene.clearColor = new Color4(0, 0, 0, 0);
      const camera = new UniversalCamera(
        "arena-camera",
        new Vector3(0, 5.4, -13.5),
        scene,
      );
      camera.setTarget(new Vector3(0, 1.8, 0.8));
      camera.fov = 0.66;
      camera.inputs.clear();
      const sky = new HemisphericLight(
        "arena-fill",
        new Vector3(0, 1, 0),
        scene,
      );
      sky.diffuse = Color3.FromHexString(cssArenaColor("fill"));
      sky.groundColor = Color3.FromHexString(cssArenaColor("void"));
      sky.intensity = 1.35;
      const playerLight = new DirectionalLight(
        "player-key",
        new Vector3(-0.35, -0.7, 0.55),
        scene,
      );
      playerLight.diffuse = Color3.FromHexString(cssArenaColor("cyan"));
      playerLight.intensity = 2.1;
      const bossLight = new DirectionalLight(
        "boss-rim",
        new Vector3(0.4, -0.45, -0.7),
        scene,
      );
      bossLight.diffuse = Color3.FromHexString(cssArenaColor("violet"));
      bossLight.intensity = 2.3;
      const world = buildArenaWorld(scene);
      const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      engine.runRenderLoop(() => scene?.render());
      this.engine = engine;
      this.scene = scene;
      this.camera = camera;
      this.world = world;
      this.reducedMotion = reducedMotion;
    } catch (error) {
      scene?.dispose();
      engine.dispose();
      throw error;
    }
  }

  async load(
    onProgress: (percent: number, stage: string) => void,
  ): Promise<void> {
    onProgress(18, "Building the ruined threshold…");
    const result = await SceneLoader.ImportMeshAsync(
      "",
      "/assets/arena/models/",
      "quaternius-demon.glb",
      this.scene,
      (event) => {
        const progress = event.lengthComputable
          ? event.loaded / Math.max(1, event.total)
          : 0.5;
        onProgress(22 + progress * 62, "Importing the demon and its clips…");
      },
    );
    const root = result.meshes[0];
    if (!root) throw new Error("Arena boss GLB has no root mesh");
    const bounds = root.getHierarchyBoundingVectors(true);
    const height = Math.max(0.01, bounds.max.y - bounds.min.y);
    const scale = 2.8 / height;
    root.scaling.setAll(scale);
    root.position = new Vector3(0, -bounds.min.y * scale + 0.95, 4.2);
    root.rotation.y = Math.PI;
    for (const group of result.animationGroups) {
      this.bossAnimations.set(group.name, group);
    }
    for (const required of Object.values(bossClips)) {
      if (!this.bossAnimations.has(required)) {
        throw new Error(`Arena boss is missing animation ${required}`);
      }
    }
    this.loaded = true;
    this.playBoss("intro", false);
    onProgress(100, "Arena ready");
  }

  update(encounter: ArenaEncounter, state: ArenaRunState, time: number): void {
    const presentation = deriveArenaPresentation(encounter, state, time);
    this.world.setPositionStates(presentation.positions);
    this.world.setAttack(presentation.activeAttack);
    this.positionPerformer(state, time);
    this.animatePerformer(time);
    const pulse = presentation.beat.downbeat ? 1.12 : 1.06;
    this.world.player.focus.scaling.setAll(
      this.reducedMotion
        ? 1
        : 1 + (1 - presentation.beat.progress) * (pulse - 1),
    );
    const attack = presentation.activeAttack;
    if (!attack) {
      this.playBoss("idle", true);
    } else if (attack.phase === "prepare") {
      this.playBoss(`${attack.type}-prepare`, true);
    } else if (attack.phase === "impact") {
      this.playBoss(`${attack.type}-impact`, false);
    } else {
      this.playBoss("opening", false);
    }
    if (state.phase === "victory") this.playBoss("defeat", false);
    if (state.phase === "failed-resolve") this.playBoss("phase", false);
    this.updateCamera(time);
  }

  playEffect(effect: ArenaEffect, songTime: number): void {
    switch (effect.type) {
      case "perform-contact":
        this.action = "perform";
        this.actionUntil = songTime + 0.22;
        this.playBoss("hit", false);
        break;
      case "boss-impact":
        this.action = effect.avoided ? "idle" : "hit";
        this.actionUntil = songTime + 0.3;
        if (!this.reducedMotion) this.cameraImpulseUntil = songTime + 0.18;
        break;
      case "outcome":
        if (effect.outcome === "victory") this.playBoss("defeat", false);
        break;
      case "input-ack":
      case "perform-flub":
      case "phrase-miss":
      case "move-start":
      case "move-arrive":
      case "boundary":
      case "move-unavailable":
      case "boss-prepare":
      case "boss-opening":
        break;
    }
  }

  resize(): void {
    this.engine.resize();
  }

  setPaused(paused: boolean): void {
    for (const animation of this.bossAnimations.values()) {
      animation.speedRatio = paused ? 0 : 1;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.engine.dispose();
  }

  private playBoss(semantic: BossSemanticClip, loop: boolean): void {
    if (!this.loaded) return;
    const source = bossClips[semantic];
    if (this.currentBossClip === source) return;
    for (const animation of this.bossAnimations.values()) animation.stop();
    this.bossAnimations
      .get(source)
      ?.start(loop, 1, undefined, undefined, false);
    this.currentBossClip = source;
  }

  private positionPerformer(state: ArenaRunState, time: number): void {
    const travel = state.travel;
    if (!travel) {
      this.world.player.root.position.copyFrom(anchorPosition[state.position]);
      return;
    }
    const progress = Math.max(
      0,
      Math.min(1, (time - travel.start) / (travel.end - travel.start)),
    );
    Vector3.LerpToRef(
      anchorPosition[travel.from],
      anchorPosition[travel.to],
      this.reducedMotion ? Math.round(progress) : progress,
      this.world.player.root.position,
    );
  }

  private animatePerformer(time: number): void {
    const active = time < this.actionUntil ? this.action : "idle";
    const beat = Math.sin(time * Math.PI * 2) * 0.04;
    this.world.player.root.rotation.y = Math.PI;
    this.world.player.root.position.y = this.reducedMotion
      ? 0
      : Math.max(0, beat);
    const strike =
      active === "perform" ? -1.25 : active === "hit" ? 0.72 : -0.18;
    this.world.player.leftArm.rotation.x = strike;
    this.world.player.rightArm.rotation.x = strike * 0.82;
    this.world.player.ward.scaling.setAll(active === "hit" ? 1.08 : 1);
  }

  private updateCamera(time: number): void {
    const impulse =
      time < this.cameraImpulseUntil
        ? Math.sin((this.cameraImpulseUntil - time) * 90) * 0.04
        : 0;
    this.camera.position.x = impulse;
  }
}
