Audio Notes

Chats:

Music processing midi stuff

Prompts:

Dark high-energy K-pop, EDM + trap bass, soaring layered vocals, distorted guitar—epic, cinematic, supernatural, nonstop battle-ready wall of sound, glamorous & dangerous.

Dark high-energy K-pop, EDM + fried 808 bass, soaring layered male vocals, distorted guitar—epic, cinematic, supernatural, nonstop battle-ready wall of sound, glamorous & dangerous.”

Dark high-energy, K-pop with EDM, trap bass, subtle Christian lyrics, soaring vocals, distorted guitar—epic, cinematic, supernatural, battle-ready, glamorous & dangerous.

Dark high-energy, K-pop with EDM, trap bass, subtle Christian lyrics, soaring male vocals, distorted guitar—epic, cinematic, supernatural, battle-ready, glamorous & dangerous.

Convert stems to midi

https://www.celemony.com/en/trial - $300+

Samplab? - $12 per mo

Rip X - $100-200

NeuralNote works for tonal instruments and is OSS!

This uses UV5R. Can we find a cli that works for split?

https://github.com/Anjok07/ultimatevocalremovergui

This splits into stems with the Demucs model. Works great!

There’s the demucs python lib here: https://github.com/facebookresearch/demucs?tab=readme-ov-file

Yea, trying this to get the drum track

demucs docker image - couldn’t get it to work easily.

nomadkaraoke/python-audio-separator (UVR5 cli)

docker run -it -v pwd:/workdir beveradb/audio-separator --model_filename htdemucs.yaml --single_stem drums input.wav

basic-pitch can be used on vocals, bass and guitar (tone instruments) to get midi

Can we simplify the midi by compressing note ranges and durations to make it simpler for kids to hit keys?

The --save-note-events will save predicted notes as csv file.

The actual quality is pretty poor though. Notes in the wrong place, etc.

aubio may be able to find notes for conversion to midi, but not sure.

Aubio ain’t working to get the midis or beat times. It’s all a mess.

ffmpeg kinda works for splitting out different drum parts

https://github.com/sonic-visualiser/sonic-annotator  ?

Using for drums only
Trying to use this with the https://vamp-plugins.org/pack.html  plugin pack. Manually installing plugins did NOT work.

Use qm note onset transform with “energy” function type and sensitivity of 25%

Use midicsv and csvmidi to grep out “Pitch_bend_c” events to make the midi cleaner (though I don’t know if it matters since it’s a separate event).

sonic-analyzer - plugin beatroot beat tracker: beats correctly plotted out the beats from the drums stem (that came from suno). We could quantize to these beats?

Docker image for sonic annotator

Looking at pyin plugin for vocal notes.



Model Notes

Figuro online model editor

nlevel.ai - nah

rodin ? stucks

https://hitem3d.ai/  actually works well I think, but WAY too many faces and vertices

Actually, when imported into roblox it’s worse than meshy

meshy works well

Game Design

We could have the battle maps be like ruined ancient temples or craggy Mountain tops or a swamp or an ice field.

Players could have to run between spots on the map to get different positions for music bonuses. The enemy boss could knock players off this positions and then we would cut their sound out of the mix and they'd have to get back. Maybe the closer positions would give a bigger bonus but be more risky for hits.

This would be harder on mobile. Would they tap to move?

If a player gets knocked down they have to tap quick over and over to revive.

We should do random loot drops after boss battles. Bosses should have different attacks and defenses.

Can we procedurally make bosses defeated 75% of the time? Is that a matter of balance or just tweaking the numbers live.

The player drops him that could get dropped right into a little "we need you"! Intro and then into a boss battle that's guided.

there should be a dangling thread at the end of each boss battle. Like a clue they need to get to the next one.

Though I don't know what the loop would be. Maybe you have to craft something with your instruments, or unlock a song?

We can have special bosses at certain days and times, with countdowns. (Retention and engagement)

Can we have instruments that can be used for healing or defense?

Or maybe a run of good notes gives you a buff for a while

Boss Ideas

Large stone giant

Could spawn by dropping from the sky, shaking the ground

Whispy siren type

Could appear by opening a portal on the ground (black/stars with blue or purple beam towards center)

Item Ideas

_Search on _fab.com

Guitar pedals

Kick pedals

Drum sticks

Guitar picks

Microphone

Mic stand

Instrument cable

Guitar, Bass, Drums, Keyboard
