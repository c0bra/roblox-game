import { UniversalCamera } from "@babylonjs/core/Cameras/universalCamera";
import { Engine } from "@babylonjs/core/Engines/engine";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { CreateCylinder } from "@babylonjs/core/Meshes/Builders/cylinderBuilder";
import { CreateSphere } from "@babylonjs/core/Meshes/Builders/sphereBuilder";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { Scene } from "@babylonjs/core/scene";
import {
  type BackdropViewState,
  backdropPresentationForState,
  iceArenaLayout,
  iceFloorTextureUrls,
  icePanoramaUrl,
} from "./backdrop-preview";
import "./backdrop-preview.css";

type BackdropViewerElements = {
  readonly canvas: HTMLCanvasElement;
  readonly resetButton: HTMLButtonElement;
  readonly root: HTMLElement;
  readonly status: HTMLElement;
};

export class IceBackdropViewer {
  private readonly camera: UniversalCamera;
  private readonly engine: Engine;
  private readonly resizeObserver: ResizeObserver;
  private readonly scene: Scene;

  constructor(private readonly elements: BackdropViewerElements) {
    this.engine = new Engine(elements.canvas, true, {
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
      stencil: false,
    });
    this.scene = new Scene(this.engine);
    this.scene.clearColor = new Color4(0.02, 0.03, 0.06, 1);
    this.camera = new UniversalCamera(
      "backdrop-camera",
      new Vector3(
        iceArenaLayout.cameraPosition.x,
        iceArenaLayout.cameraPosition.y,
        iceArenaLayout.cameraPosition.z,
      ),
      this.scene,
    );
    this.camera.fov = 1.08;
    this.camera.minZ = 0.01;
    this.camera.angularSensibility = 2600;
    this.camera.inertia = 0.72;
    this.camera.inputs.removeByType("FreeCameraKeyboardMoveInput");
    this.camera.attachControl(elements.canvas, false);
    this.resetView();

    this.buildHemisphere();
    this.buildArenaFloor();
    this.scene.onBeforeRenderObservable.add(() => {
      this.camera.rotation.x = Math.max(
        -0.68,
        Math.min(1.12, this.camera.rotation.x),
      );
    });
    this.engine.runRenderLoop(() => this.scene.render());

    this.resizeObserver = new ResizeObserver(() => this.engine.resize());
    this.resizeObserver.observe(elements.canvas);
    elements.resetButton.addEventListener("click", this.resetView);
  }

  dispose(): void {
    this.elements.resetButton.removeEventListener("click", this.resetView);
    this.resizeObserver.disconnect();
    this.camera.detachControl();
    this.engine.dispose();
  }

  private buildHemisphere(): void {
    const material = new StandardMaterial("ice-panorama-material", this.scene);
    const texture = new Texture(icePanoramaUrl, this.scene, {
      invertY: false,
      noMipmap: false,
      onError: () => this.setViewState("error"),
      onLoad: () => this.revealWhenRendered(),
      samplingMode: Texture.TRILINEAR_SAMPLINGMODE,
    });
    texture.uScale = 1;
    texture.wrapU = Texture.WRAP_ADDRESSMODE;
    texture.wrapV = Texture.CLAMP_ADDRESSMODE;
    material.backFaceCulling = false;
    material.diffuseTexture = texture;
    material.disableLighting = true;
    material.emissiveColor = Color3.White();
    material.emissiveTexture = texture;

    const hemisphere = CreateSphere(
      "ice-panorama-hemisphere",
      {
        diameter: 120,
        segments: 64,
        sideOrientation: Mesh.BACKSIDE,
        slice: 0.82,
      },
      this.scene,
    );
    hemisphere.material = material;
    hemisphere.position.set(
      iceArenaLayout.cameraPosition.x,
      iceArenaLayout.cameraPosition.y,
      iceArenaLayout.cameraPosition.z,
    );
    hemisphere.rotation.y = 0;
  }

  private revealWhenRendered(): void {
    this.scene.executeWhenReady(() => {
      this.scene.onAfterRenderObservable.addOnce(() => {
        this.setViewState("ready");
      });
    });
  }

  private buildArenaFloor(): void {
    const styles = getComputedStyle(document.documentElement);
    const floorColor = Color3.FromHexString(
      styles.getPropertyValue(iceArenaLayout.floorColorToken).trim(),
    );
    const energyColor = Color3.FromHexString(
      styles.getPropertyValue("--cyan").trim(),
    );
    const light = new HemisphericLight(
      "ice-arena-light",
      Vector3.Up(),
      this.scene,
    );
    light.diffuse = Color3.White();
    light.groundColor = floorColor.scale(0.28);
    light.intensity = 0.86;

    const albedoTexture = this.buildFloorTexture(iceFloorTextureUrls.albedo);
    const normalTexture = this.buildFloorTexture(iceFloorTextureUrls.normal);
    const roughnessTexture = this.buildFloorTexture(
      iceFloorTextureUrls.roughness,
    );
    normalTexture.gammaSpace = false;
    normalTexture.level = 0.18;
    roughnessTexture.gammaSpace = false;

    const floorMaterial = new PBRMaterial(
      "ice-arena-floor-material",
      this.scene,
    );
    floorMaterial.albedoColor = floorColor;
    floorMaterial.albedoTexture = albedoTexture;
    floorMaterial.bumpTexture = normalTexture;
    floorMaterial.emissiveColor = floorColor.scale(0.06);
    floorMaterial.metallic = 0;
    floorMaterial.metallicTexture = roughnessTexture;
    floorMaterial.roughness = 1;
    floorMaterial.useMetallnessFromMetallicTextureBlue = false;
    floorMaterial.useRoughnessFromMetallicTextureAlpha = false;
    floorMaterial.useRoughnessFromMetallicTextureGreen = true;

    const rimMaterial = new StandardMaterial(
      "ice-arena-rim-material",
      this.scene,
    );
    rimMaterial.disableLighting = true;
    rimMaterial.emissiveColor = energyColor.scale(0.72);

    const rim = CreateCylinder(
      "ice-arena-rim",
      {
        diameter: iceArenaLayout.floorDiameter + 0.42,
        height: 0.12,
        tessellation: 96,
      },
      this.scene,
    );
    rim.material = rimMaterial;
    rim.position.set(
      iceArenaLayout.floorCenter.x,
      iceArenaLayout.floorCenter.y - iceArenaLayout.floorHeight / 2 + 0.02,
      iceArenaLayout.floorCenter.z,
    );

    const floor = CreateCylinder(
      "ice-arena-floor",
      {
        diameter: iceArenaLayout.floorDiameter,
        height: iceArenaLayout.floorHeight,
        tessellation: 96,
      },
      this.scene,
    );
    floor.material = floorMaterial;
    floor.position.set(
      iceArenaLayout.floorCenter.x,
      iceArenaLayout.floorCenter.y,
      iceArenaLayout.floorCenter.z,
    );
  }

  private buildFloorTexture(url: string): Texture {
    const texture = new Texture(url, this.scene, {
      invertY: false,
      noMipmap: false,
      samplingMode: Texture.TRILINEAR_SAMPLINGMODE,
    });
    texture.uScale = 3;
    texture.vScale = 3;
    texture.wrapU = Texture.WRAP_ADDRESSMODE;
    texture.wrapV = Texture.WRAP_ADDRESSMODE;
    return texture;
  }

  private setViewState(state: BackdropViewState): void {
    const presentation = backdropPresentationForState(state);
    this.elements.root.dataset.state = presentation.rootState;
    this.elements.status.textContent = presentation.status;
  }

  private readonly resetView = (): void => {
    this.camera.rotation.set(0.04, Math.PI, 0);
  };
}
