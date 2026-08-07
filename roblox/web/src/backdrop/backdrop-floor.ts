import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";
import { DynamicTexture } from "@babylonjs/core/Materials/Textures/dynamicTexture";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { CreateCylinder } from "@babylonjs/core/Meshes/Builders/cylinderBuilder";
import { Scene } from "@babylonjs/core/scene";
import { iceArenaLayout, iceFloorTextureUrls } from "./backdrop-preview";

const floorFadeTextureSize = 256;

export const buildBackdropFloor = (scene: Scene): void => {
  const styles = getComputedStyle(document.documentElement);
  const floorColor = Color3.FromHexString(
    styles.getPropertyValue(iceArenaLayout.floorColorToken).trim(),
  );
  const hazeColor = Color3.FromHexString(
    styles.getPropertyValue(iceArenaLayout.hazeColorToken).trim(),
  );
  scene.fogMode = Scene.FOGMODE_LINEAR;
  scene.fogColor = hazeColor;
  scene.fogStart = iceArenaLayout.fogStart;
  scene.fogEnd = iceArenaLayout.fogEnd;

  const light = new HemisphericLight("ice-arena-light", Vector3.Up(), scene);
  light.diffuse = Color3.White();
  light.groundColor = floorColor.scale(0.28);
  light.intensity = 0.86;

  const albedoTexture = buildFloorTexture(scene, iceFloorTextureUrls.albedo);
  const normalTexture = buildFloorTexture(scene, iceFloorTextureUrls.normal);
  const roughnessTexture = buildFloorTexture(
    scene,
    iceFloorTextureUrls.roughness,
  );
  normalTexture.gammaSpace = false;
  normalTexture.level = 0.18;
  roughnessTexture.gammaSpace = false;

  const floorMaterial = new PBRMaterial("ice-arena-floor-material", scene);
  floorMaterial.albedoColor = floorColor.scale(iceArenaLayout.albedoLift);
  floorMaterial.albedoTexture = albedoTexture;
  floorMaterial.bumpTexture = normalTexture;
  floorMaterial.emissiveColor = floorColor.scale(
    iceArenaLayout.emissiveStrength,
  );
  floorMaterial.metallic = 0;
  floorMaterial.metallicTexture = roughnessTexture;
  floorMaterial.opacityTexture = buildFloorOpacityTexture(scene);
  floorMaterial.roughness = 1;
  floorMaterial.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
  floorMaterial.useMetallnessFromMetallicTextureBlue = false;
  floorMaterial.useRoughnessFromMetallicTextureAlpha = false;
  floorMaterial.useRoughnessFromMetallicTextureGreen = true;
  floorMaterial.useSpecularOverAlpha = false;

  const floor = CreateCylinder(
    "ice-arena-floor",
    {
      diameter: iceArenaLayout.surfaceDiameter,
      height: iceArenaLayout.floorHeight,
      tessellation: 96,
    },
    scene,
  );
  floor.material = floorMaterial;
  floor.position.set(
    iceArenaLayout.floorCenter.x,
    iceArenaLayout.floorCenter.y,
    iceArenaLayout.floorCenter.z,
  );
};

const buildFloorTexture = (scene: Scene, url: string): Texture => {
  const texture = new Texture(url, scene, {
    invertY: false,
    noMipmap: false,
    samplingMode: Texture.TRILINEAR_SAMPLINGMODE,
  });
  const repeat =
    iceArenaLayout.surfaceDiameter / iceArenaLayout.textureWorldSize;
  texture.uScale = repeat;
  texture.vScale = repeat;
  texture.wrapU = Texture.WRAP_ADDRESSMODE;
  texture.wrapV = Texture.WRAP_ADDRESSMODE;
  return texture;
};

const buildFloorOpacityTexture = (scene: Scene): DynamicTexture => {
  const texture = new DynamicTexture(
    "ice-floor-opacity",
    { width: floorFadeTextureSize, height: floorFadeTextureSize },
    scene,
    false,
    Texture.BILINEAR_SAMPLINGMODE,
  );
  const context = texture.getContext();
  const radius = floorFadeTextureSize / 2;
  const gradient = context.createRadialGradient(
    radius,
    radius,
    radius * iceArenaLayout.opacityFadeStart,
    radius,
    radius,
    radius * iceArenaLayout.opacityFadeEnd,
  );
  gradient.addColorStop(0, "white");
  gradient.addColorStop(1, "black");
  context.fillStyle = gradient;
  context.fillRect(0, 0, floorFadeTextureSize, floorFadeTextureSize);
  texture.gammaSpace = false;
  texture.getAlphaFromRGB = true;
  texture.wrapU = Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  texture.update(false);
  return texture;
};
