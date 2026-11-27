# Local song editor

Open a prepared song project from the repository root.

```sh
(cd tools/chart-pipeline && bun install --frozen-lockfile)
(cd tools/content-authoring && bun install --frozen-lockfile)
bun tools/content-authoring/server.ts content/projects/blackened-crown
```

The terminal prints a loopback address. The project selector lists prepared sibling folders containing `project.json`. Pass another folder on the command line to open a project elsewhere. Set `AUTHORING_PORT` for a repeatable address. The editor uses the installed FFmpeg and the chart pipeline's existing analysis tools and Vamp plugins.

For read/write Wi-Fi access, set `AUTHORING_WIFI_ADDRESS` to the Mac’s private Wi-Fi IPv4 address and `AUTHORING_WIFI_KEY` to a random hex string of at least 32 characters. The phone URL is `http://<address>:<port>/<key>/`. It opens the chosen project and shares its drafts and analysis jobs with the desktop editor. The link has no expiry. Anyone on the reachable network with the link can edit. Keep the key out of version control.

## Listen and correct

1. Choose a passage and press **Play**. The overview covers the full master. Click or scrub it to seek. A named passage sets In and Out in seconds and enables looping. Click **Loop** again to listen past its boundary.
2. Click a layer name to edit its chart. **S** solos its recording, **M** mutes it and **Full mix** returns to the master. The roll displays the selected stem waveform, three note lanes and the playhead. **Hear chart taps** auditions the selected difficulty against the recording. When **Compare notes** shows a generated chart or previous save, the taps follow that comparison. Select **Current draft** to hear your saved work again.
3. Select a note or use **Next note**. Adjust its start, hold length or lane and press **Apply edit**. **Add note** inserts a tap at the playhead. **Mark selected layer resting** cuts notes across every difficulty in the In–Out range and retains a rest marker for subsequent chart imports. Remove a marker with its × control. Undo reverses the most recent edit.
4. Use **Save passage** with an existing name to update its boundaries. **Save draft** persists the chart, passages, rests, clock, encounter entries and listening notes together. Compare notes can overlay the previous save or the latest generated chart against the current roll.

**Blackened Crown** opens with four layers and twelve charts. The 2:35–2:45 breakdown bookmark includes the last drum strike at 2:39.909, its decay, and the first returning strike at 2:43.108. The guitar outro bookmark begins at the final detected drum beat and includes the decay to the end of the master. Both bookmarks are listening ranges.

## Analysis and review

**Beat detection** runs BeatRoot on the drum stem. **Transcription and charts** uses the existing pitch, onset, quantization and difficulty processors. Each instrument has a second pass for missing notes. In **Compare notes**, choose the generated Normal chart to see clickable findings for active passages that still lack a note or hold. Marked rests also apply to the generated preview. Chart derivation uses the saved beat grid when present, or the detected grid for a new project. Editing the grid makes incompatible chart suggestions unavailable until a rerun. The author can edit individual beats and the number of beats per bar. Detecting beats again leaves edits intact.

Jobs report their active stage. Cancellation and failure retain the last successful suggestions. **Replace selected chart** explicitly replaces that one difficulty and reapplies its marked rests. Review prior manual corrections with the overlay and Undo. **Save all suggested charts** applies all difficulties, honors marked rests, and saves the project through its normal revision check. **Current draft · whole song** compares note starts and time on beats with a note or hold across roles at the selected difficulty. This is a screening measure for the Normal participation guideline in [the balance framework](../../BALANCE_FRAMEWORK.md#participation-and-re-entry). Use the detected clock until a draft clock exists. **Passage review** shows note counts for every layer and difficulty. Fixed and conditional entries use the In–Out range, warning time and layer scope. Click a review finding to seek to it.

A project may provide `transcription-fills.json` with measured source events for an otherwise empty span. The editor checks the source revision and runs those events through the difficulty compiler on the current grid. Existing chart notes or a hold crossing into a span take precedence. Blackened Crown uses this for bass and the Other/guitar layer in its breakdown review loop.

## Files and exports

The editor reads full-length audio from the paths in `project.json`. Chart times include the manifest's clip offset. The original chart bundle remains available for comparison. A project can use different playable names for the four legacy separator outputs.

`authoring-draft.json` is the saved draft. The editor keeps the preceding save in `authoring-draft.json.previous`, accepts the older `review-notes.txt` as initial listening notes, and stores reusable processor output in `analysis-suggestions.json`. A conflicting save leaves the newer disk revision intact and lets the other tab download its unsaved edits.

**Export JSON** downloads the saved draft, audio fingerprints, three validated timeline projections, per-beat activity and review findings. **Studio model** downloads a local `.rbxmx` data folder with the same revision and charts. Insert the model into a disposable place. Each child is a StringValue containing JSON. For example, read `Normal.Value` through `HttpService:JSONDecode` and pass it to the shared timeline reader. Audio asset bindings, owner approval, roster coverage and in-game playtesting belong to the staging and release workflow.

On another machine, rebuild the ignored bundle from the matching stems or master, then set `legacyBundle` to its directory. A different recording or chart source requires a new compatible draft.

```sh
./chart build --song 'Blackened Crown - New.mp3' --outro-role guitar --output build/blackened-crown-authoring
bun test tools/content-authoring
```
