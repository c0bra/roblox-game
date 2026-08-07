import { UniversalCamera } from "@babylonjs/core/Cameras/universalCamera";
import { Engine } from "@babylonjs/core/Engines/engine";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { CreateSphere } from "@babylonjs/core/Meshes/Builders/sphereBuilder";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { Scene } from "@babylonjs/core/scene";
import { buildBackdropEnvironment } from "./backdrop-environment";
import { buildBackdropFloor } from "./backdrop-floor";
import { clampBackdropPosition } from "./backdrop-movement";
import {
  type BackdropViewState,
  backdropPresentationForState,
  backdropRenderScaleForDevicePixelRatio,
  iceArenaLayout,
  icePanoramaUrlForProfile,
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
  private readonly hemisphere: Mesh;
  private readonly resizeObserver: ResizeObserver;
  private readonly scene: Scene;

  constructor(private readonly elements: BackdropViewerElements) {
    this.engine = new Engine(
      elements.canvas,
      true,
      {
        powerPreference: "high-performance",
        preserveDrawingBuffer: false,
        stencil: false,
      },
      true,
    );
    this.engine.setHardwareScalingLevel(
      1 / backdropRenderScaleForDevicePixelRatio(window.devicePixelRatio),
    );
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
    this.camera.fov = iceArenaLayout.cameraFov;
    this.camera.minZ = 0.01;
    this.camera.speed = iceArenaLayout.cameraSpeed;
    this.camera.angularSensibility = 2600;
    this.camera.inertia = 0.72;
    this.camera.keysUp = [87, 38];
    this.camera.keysDown = [83, 40];
    this.camera.keysLeft = [65, 37];
    this.camera.keysRight = [68, 39];
    this.camera.attachControl(elements.canvas, false);

    this.hemisphere = this.buildHemisphere();
    buildBackdropFloor(this.scene);
    buildBackdropEnvironment(this.scene);
    this.resetView();
    this.camera.onAfterCheckInputsObservable.add(() => {
      this.constrainCamera();
    });
    this.scene.onBeforeRenderObservable.add(() => {
      this.camera.rotation.x = Math.max(
        -0.68,
        Math.min(1.12, this.camera.rotation.x),
      );
    });
    this.engine.runRenderLoop(() => this.scene.render());

    this.resizeObserver = new ResizeObserver(() => {
      this.engine.setHardwareScalingLevel(
        1 / backdropRenderScaleForDevicePixelRatio(window.devicePixelRatio),
      );
      this.engine.resize();
    });
    this.resizeObserver.observe(elements.canvas);
    elements.resetButton.addEventListener("click", this.resetView);
  }

  dispose(): void {
    this.elements.resetButton.removeEventListener("click", this.resetView);
    this.resizeObserver.disconnect();
    this.camera.detachControl();
    this.engine.dispose();
  }

  private buildHemisphere(): Mesh {
    const material = new StandardMaterial("ice-panorama-material", this.scene);
    const panoramaUrl = icePanoramaUrlForProfile({
      devicePixelRatio: window.devicePixelRatio,
      maxTextureSize: this.engine.getCaps().maxTextureSize,
      viewportWidth: this.elements.canvas.clientWidth,
    });
    const texture = new Texture(panoramaUrl, this.scene, {
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
    material.fogEnabled = false;

    const hemisphere = CreateSphere(
      "ice-panorama-hemisphere",
      {
        diameter: 120,
        segments: 64,
        sideOrientation: Mesh.BACKSIDE,
        slice: iceArenaLayout.panoramaSphereSlice,
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
    return hemisphere;
  }

  private constrainCamera(): void {
    const position = clampBackdropPosition(
      { x: this.camera.position.x, z: this.camera.position.z },
      iceArenaLayout.cameraTravelRadius,
    );
    this.camera.position.set(
      position.x,
      iceArenaLayout.cameraPosition.y,
      position.z,
    );
    this.hemisphere.position.set(
      position.x,
      iceArenaLayout.cameraPosition.y,
      position.z,
    );
  }

  private revealWhenRendered(): void {
    this.scene.executeWhenReady(() => {
      this.scene.onAfterRenderObservable.addOnce(() => {
        this.setViewState("ready");
      });
    });
  }

  private setViewState(state: BackdropViewState): void {
    const presentation = backdropPresentationForState(state);
    this.elements.root.dataset.state = presentation.rootState;
    this.elements.status.textContent = presentation.status;
  }

  private readonly resetView = (): void => {
    this.camera.cameraDirection.setAll(0);
    this.camera.cameraRotation.setAll(0);
    this.camera.position.set(
      iceArenaLayout.cameraPosition.x,
      iceArenaLayout.cameraPosition.y,
      iceArenaLayout.cameraPosition.z,
    );
    this.camera.rotation.set(0.04, Math.PI, 0);
    this.constrainCamera();
  };
}
