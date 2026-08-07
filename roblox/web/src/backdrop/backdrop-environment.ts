import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { CreateCylinder } from "@babylonjs/core/Meshes/Builders/cylinderBuilder";
import { CreateIcoSphere } from "@babylonjs/core/Meshes/Builders/icoSphereBuilder";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import type { Scene } from "@babylonjs/core/scene";
import {
  type IcePerimeterCluster,
  icePerimeterClusters,
} from "./backdrop-environment-layout";
import { iceArenaLayout } from "./backdrop-preview";

type IceEnvironmentMaterials = {
  readonly rock: PBRMaterial;
  readonly snow: PBRMaterial;
  readonly crystal: StandardMaterial;
};

type ClusterBuildContext = {
  readonly floorTop: number;
  readonly materials: IceEnvironmentMaterials;
  readonly scene: Scene;
};

const buildMaterials = (scene: Scene): IceEnvironmentMaterials => {
  const styles = getComputedStyle(document.documentElement);
  const floor = Color3.FromHexString(
    styles.getPropertyValue(iceArenaLayout.floorColorToken).trim(),
  );
  const haze = Color3.FromHexString(
    styles.getPropertyValue(iceArenaLayout.hazeColorToken).trim(),
  );
  const panel = Color3.FromHexString(
    styles.getPropertyValue("--panel-strong").trim(),
  );
  const cyan = Color3.FromHexString(styles.getPropertyValue("--cyan").trim());

  const rock = new PBRMaterial("ice-perimeter-rock", scene);
  rock.albedoColor = Color3.Lerp(floor, panel, 0.74);
  rock.metallic = 0;
  rock.roughness = 0.88;

  const snow = new PBRMaterial("ice-perimeter-snow", scene);
  snow.albedoColor = Color3.Lerp(haze, Color3.White(), 0.32);
  snow.metallic = 0;
  snow.roughness = 0.96;

  const crystal = new StandardMaterial("ice-perimeter-crystal", scene);
  crystal.alpha = 0.78;
  crystal.backFaceCulling = false;
  crystal.diffuseColor = Color3.Lerp(floor, cyan, 0.52);
  crystal.emissiveColor = cyan.scale(0.08);
  crystal.specularColor = Color3.White().scale(0.72);
  return { rock, snow, crystal };
};

const buildCluster = (
  context: ClusterBuildContext,
  cluster: IcePerimeterCluster,
  index: number,
): void => {
  const root = new TransformNode(`ice-cluster-${index}`, context.scene);
  root.position.set(
    Math.cos(cluster.angle) * cluster.radius,
    context.floorTop,
    Math.sin(cluster.angle) * cluster.radius,
  );
  root.rotation.y = cluster.angle * 1.7;

  const rock = CreateIcoSphere(
    `ice-cluster-${index}-rock`,
    { radius: 1, subdivisions: 1, flat: true },
    context.scene,
  );
  rock.parent = root;
  rock.scaling = Vector3.FromArray(cluster.rockScale);
  rock.position.y = cluster.rockScale[1] * 0.72;
  rock.rotation.set(0.08 * (index % 3), 0.24 * index, 0.11 * (index % 2));
  rock.material = context.materials.rock;
  rock.isPickable = false;

  const shoulder = CreateIcoSphere(
    `ice-cluster-${index}-shoulder`,
    { radius: 1, subdivisions: 1, flat: true },
    context.scene,
  );
  shoulder.parent = root;
  shoulder.scaling.set(
    cluster.rockScale[0] * 0.62,
    cluster.rockScale[1] * 0.48,
    cluster.rockScale[2] * 0.7,
  );
  shoulder.position.set(
    cluster.spread,
    cluster.rockScale[1] * 0.38,
    -cluster.spread * 0.24,
  );
  shoulder.rotation.y = -0.38 * index;
  shoulder.material = context.materials.rock;
  shoulder.isPickable = false;

  const snow = CreateIcoSphere(
    `ice-cluster-${index}-snow`,
    { radius: 1, subdivisions: 1, flat: true },
    context.scene,
  );
  snow.parent = root;
  snow.scaling.set(
    cluster.rockScale[0] * 0.68,
    cluster.rockScale[1] * 0.18,
    cluster.rockScale[2] * 0.7,
  );
  snow.position.set(-0.16, cluster.rockScale[1] * 1.38, 0.08);
  snow.material = context.materials.snow;
  snow.isPickable = false;

  if (cluster.crystalHeight === 0) return;
  const crystal = CreateCylinder(
    `ice-cluster-${index}-crystal`,
    {
      diameterBottom: cluster.crystalHeight * 0.3,
      diameterTop: 0,
      height: cluster.crystalHeight,
      tessellation: 6,
    },
    context.scene,
  );
  crystal.parent = root;
  crystal.position.set(
    -cluster.spread * 0.78,
    cluster.crystalHeight / 2,
    cluster.spread * 0.28,
  );
  crystal.rotation.z = (index % 2 === 0 ? 1 : -1) * 0.08;
  crystal.material = context.materials.crystal;
  crystal.isPickable = false;
};

export const buildBackdropEnvironment = (scene: Scene): void => {
  const context = {
    floorTop: iceArenaLayout.floorCenter.y + iceArenaLayout.floorHeight / 2,
    materials: buildMaterials(scene),
    scene,
  } satisfies ClusterBuildContext;
  icePerimeterClusters.forEach((cluster, index) => {
    buildCluster(context, cluster, index);
  });
};
