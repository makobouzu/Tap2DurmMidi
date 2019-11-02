import copy
import os
base = os.path.dirname(os.path.abspath(__file__))


# Magenta specific stuff
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from magenta import music as mm
from magenta.music import midi_synth
from magenta.music import midi_io


# Load some configs to be used later
dc_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity'].data_converter

# load model
GROOVAE_2BAR_TAP_FIXED_VELOCITY = "groovae_2bar_tap_fixed_velocity.tar"
config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
groovae_2bar_tap = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path="models/"+GROOVAE_2BAR_TAP_FIXED_VELOCITY)


# Calculate quantization steps but do not remove microtiming
def quantize(s, steps_per_quarter=4):
    return mm.sequences_lib.quantize_note_sequence(s, steps_per_quarter)


def is_4_4(s):
    ts = s.time_signatures[0]
    return (ts.numerator == 4 and ts.denominator ==4)

# Some midi files come by default from different instrument channels
# Quick and dirty way to set midi files to be recognized as drums
def set_to_drums(ns):
    for n in ns.notes:
        n.instrument=9
        n.is_drum = True


# If a sequence has notes at time before 0.0, scootch them up to 0
def start_notes_at_0(s):
    for n in s.notes:
        if n.start_time < 0:
            n.end_time -= n.start_time
            n.start_time = 0
    return s


# quickly change the tempo of a midi sequence and adjust all notes
def change_tempo(note_sequence, new_tempo):
    new_sequence = copy.deepcopy(note_sequence)
    ratio = note_sequence.tempos[0].qpm / new_tempo
    for note in new_sequence.notes:
        note.start_time = note.start_time * ratio
        note.end_time = note.end_time * ratio
    new_sequence.tempos[0].qpm = new_tempo
    return new_sequence


# quick method for turning a drumbeat into a tapped rhythm
def get_tapped_2bar(s, velocity=85, ride=False):
    new_s = dc_tap.to_notesequences(dc_tap.to_tensors(s).inputs)[0]
    new_s = change_tempo(new_s, s.tempos[0].qpm)
    if velocity != 0:
        for n in new_s.notes:
            n.velocity = velocity
    if ride:
        for n in new_s.notes:
            n.pitch = 4
    return new_s


def drumify(s, model, temperature=1.0):
    encoding, mu, sigma = model.encode([s])
    decoded = model.decode(encoding, length=32, temperature=temperature)
    return decoded[0]


def main():
    #load midi file
    loaded_sequence = mm.midi_file_to_note_sequence(base+"/input/input.mid")
    
    s = loaded_sequence
    s = change_tempo(get_tapped_2bar(s, velocity=85, ride=False), s.tempos[0].qpm)

    h = drumify(s, groovae_2bar_tap)
    h = change_tempo(h, s.tempos[0].qpm)
        
    midi_io.note_sequence_to_midi_file(start_notes_at_0(h), base+"/output/output.mid")

    print("Generate Done")

if __name__ == "__main__":
    main()

