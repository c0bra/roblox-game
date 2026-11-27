// Roblox XML models support StringValue payloads without executing code when inserted.
export function studioModel(chunks: Record<string, unknown>) {
	const escape = (s: string) =>
		s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
	const items = Object.entries(chunks).map(
		([key, value], i) =>
			`<Item class="StringValue" referent="RBX${i + 1}"><Properties><string name="Name">${escape(key)}</string><string name="Value">${escape(JSON.stringify(value))}</string></Properties></Item>`,
	);
	return `<roblox version="4"><Item class="Folder" referent="RBX0"><Properties><string name="Name">SongAuthoringDraft</string></Properties>${items.join("")}</Item></roblox>`;
}
