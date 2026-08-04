import { describe, expect, test } from "bun:test";
import { levelCatalog, levelCatalogSchema } from "../src/data/level-catalog";
import { appShell } from "../src/ui/templates";

describe("selection screen template", () => {
  test("Given two catalog levels, when rendered, then both are selectable", () => {
    const html = appShell(levelCatalog);
    expect(html).toContain('data-level="heavens-edge"');
    expect(html).toContain("Heaven&#39;s Edge");
    expect(html).toContain('data-level="blackened-crown"');
    expect(html).toContain("Blackened Crown");
  });

  test("Given markup in a catalog title, when rendered, then it is escaped", () => {
    const catalog = levelCatalogSchema.parse({
      defaultLevelId: "safe-song",
      levels: [{ id: "safe-song", title: '<img src=x onerror="alert(1)">' }],
    });
    const html = appShell(catalog);
    expect(html).not.toContain("<img");
    expect(html).toContain("&lt;img");
  });
});
