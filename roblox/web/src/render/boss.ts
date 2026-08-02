import "@babylonjs/loaders/glTF";
import {
  ArcRotateCamera,
  Color3,
  Color4,
  DirectionalLight,
  Engine,
  HemisphericLight,
  Mesh,
  MeshBuilder,
  PBRMaterial,
  Scene,
  SceneLoader,
  StandardMaterial,
  TransformNode,
  Vector3,
} from "@babylonjs/core";

type BossMood = "idle" | "hit" | "attack" | "defeated";

export class BossScene {
  private readonly engine: Engine;
  private readonly scene: Scene;
  private root: TransformNode | undefined;
  private mood: BossMood = "idle";
  private moodUntil = 0;
  private reducedMotion = false;

  constructor(canvas: HTMLCanvasElement) {
    this.engine = new Engine(canvas, true, {
      stencil: true,
      preserveDrawingBuffer: false,
    });
    this.scene = new Scene(this.engine);
    this.scene.clearColor = new Color4(0, 0, 0, 0);
    const camera = new ArcRotateCamera(
      "camera",
      Math.PI / 2,
      Math.PI / 2.2,
      8,
      Vector3.Zero(),
      this.scene,
    );
    camera.fov = 0.7;
    camera.inputs.clear();
    new HemisphericLight("sky", new Vector3(0, 1, 0), this.scene).intensity =
      2.1;
    const key = new DirectionalLight(
      "key",
      new Vector3(0.15, -0.25, -1),
      this.scene,
    );
    key.diffuse = Color3.FromHexString("#8cecff");
    key.intensity = 2.4;
    const rim = new DirectionalLight(
      "rim",
      new Vector3(-0.4, -0.5, 1),
      this.scene,
    );
    rim.diffuse = Color3.FromHexString("#9c6cff");
    rim.intensity = 3;
    this.reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
  }

  async load(): Promise<void> {
    try {
      const result = await SceneLoader.ImportMeshAsync(
        "",
        "/assets/",
        "three-head-ghost.glb",
        this.scene,
      );
      const root = result.meshes[0];
      if (!root) throw new Error("Boss model has no root mesh");
      this.root = root;
      root.scaling = new Vector3(1.16, 1.16, 1.16);
      root.position = new Vector3(0, -2.35, 0);
      this.tintModel();
    } catch {
      this.buildFallbackBoss();
    }
    this.engine.runRenderLoop(() => {
      this.animate();
      this.scene.render();
    });
  }

  setMood(mood: BossMood, duration = 0.55): void {
    this.mood = mood;
    this.moodUntil = performance.now() + duration * 1_000;
  }

  resize(): void {
    this.engine.resize();
  }

  dispose(): void {
    this.engine.dispose();
  }

  private animate(): void {
    const root = this.root;
    if (!root || this.reducedMotion) return;
    const now = performance.now();
    const pulse = Math.sin(now / 650) * 0.035;
    root.position.y = -2.35 + pulse;
    root.rotation.y = Math.sin(now / 1_900) * 0.08;
    if (now > this.moodUntil && this.mood !== "defeated") this.mood = "idle";
    if (this.mood === "hit") root.rotation.z = Math.sin(now / 45) * 0.09;
    else if (this.mood === "attack")
      root.scaling.setAll(1.16 + Math.max(0, Math.sin(now / 85)) * 0.08);
    else if (this.mood === "defeated") {
      root.rotation.z += 0.008;
      root.position.y -= 0.012;
    } else {
      root.rotation.z *= 0.8;
      root.scaling.setAll(1.16);
    }
  }

  private tintModel(): void {
    for (const material of this.scene.materials) {
      if (material instanceof PBRMaterial) {
        material.emissiveColor = Color3.FromHexString("#160a2c");
        material.emissiveIntensity = 0.18;
        material.metallic = 0.08;
        material.roughness = 0.72;
      }
    }
  }

  private buildFallbackBoss(): void {
    const root = new TransformNode("boss", this.scene);
    this.root = root;
    root.position.y = -1.7;
    const body = MeshBuilder.CreatePolyhedron(
      "body",
      { type: 2, size: 2.25 },
      this.scene,
    );
    body.parent = root;
    body.scaling.y = 1.25;
    const material = new StandardMaterial("void", this.scene);
    material.diffuseColor = Color3.FromHexString("#301b55");
    material.emissiveColor = Color3.FromHexString("#6e36bf");
    body.material = material;
    for (const x of [-1.15, 0, 1.15]) {
      const head = MeshBuilder.CreatePolyhedron(
        "head",
        { type: 1, size: 0.72 },
        this.scene,
      );
      head.parent = root;
      head.position = new Vector3(x, 1.5 - Math.abs(x) * 0.18, 0);
      head.material = material;
      const eye = MeshBuilder.CreateSphere(
        "eye",
        { diameter: 0.14 },
        this.scene,
      );
      eye.parent = head;
      eye.position.z = -0.6;
      const eyeMaterial = new StandardMaterial("eye-light", this.scene);
      eyeMaterial.emissiveColor = Color3.FromHexString("#d9fbff");
      eye.material = eyeMaterial;
    }
    for (const mesh of this.scene.meshes) {
      if (mesh instanceof Mesh) mesh.isPickable = false;
    }
  }
}
