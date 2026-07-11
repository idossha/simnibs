# Neuralynx Macro vs Micro Electrodes

## The 128-Channel Boundary

In the ATLAS / Digital Lynx SX systems, channels are numbered sequentially across
hardware combo boards (32 channels each). A 4-slot Digital Lynx 4SX chassis holds
channels 0-127. Channels 128+ reside on additional board slots.

Macro and micro signals travel through **physically different front-end hardware**:

| Property | Micro channels | Macro channels |
|---|---|---|
| Front-end | CHET headstage preamps (unity-gain, high-Z) | ATLAS Headboxes (64-ch active buffers) |
| Electrode | ~40 um Pt-Ir wires | ~1.3 mm Pt-Ir cylindrical rings |
| Impedance | 100 kOhm - 2.4 MOhm | ~2-8 kOhm |
| Typical Fs | 32 kHz | 2-4 kHz |
| Referencing | 9th uninsulated wire (local, hardwired) | Flexible per-bank DRS selection |

The assignment of macro/micro to specific channel ranges depends on the physical
cabling to the combo board slots. There is no universal standard -- check NCS
file headers to determine which channels are which.

## Behnke-Fried Hybrid Electrodes (Ad-Tech)

The most common hybrid depth electrode used in clinical sEEG:

- **Outer macro contacts**: 4-12 cylindrical Pt-Ir rings (1.3 mm dia, 0.8 mm length),
  ~2 kOhm impedance, 5 mm spacing.
- **Inner micro wire bundle (IWB)**: 9 Pt-Ir wires (40 um dia) through hollow core.
  8 polyamide-insulated recording wires (100-300 kOhm) + 1 uninsulated reference wire.
  Micro wires extend 4-6 mm beyond electrode tip after insertion.

## Why Micro Channels Are Noisier

1. **Impedance mismatch**: Creates a voltage divider with the amplifier input that
   attenuates and distorts signals below 60 Hz (Nelson et al., PMC3476479).
2. **Higher thermal noise**: Johnson-Nyquist noise scales with impedance.
3. **60 Hz line noise**: SNR often < 10 dB on micro wires. Capacitive coupling from
   power lines (~11.4 uV rms) and fluorescent lights (~9.7 uV rms).
4. **Movement artifacts**: Thin wires extending past electrode tip are highly
   susceptible to brain pulsation and patient movement.
5. **Variable deployment**: Only ~3/8 micro wires per bundle typically yield good
   recordings; wire dispersion is unpredictable.

## Implications for TI Stimulation Analysis

- If macro channels are sampled at 2-4 kHz, their Nyquist frequency is 1-2 kHz --
  below the 2-9 kHz TI artifact band. These channels **cannot capture** the stimulation
  artifact.
- Micro channels at 32 kHz (Nyquist = 16 kHz) capture the full artifact.
- The "cleaner" signal on higher-numbered channels may reflect macro contacts with
  lower noise but potentially insufficient bandwidth for TI analysis.
- **Verify sampling rates from headers** to determine which channels are usable.

## NCS Header Fields to Compare

Key fields that differ between macro and micro channels:

- `SamplingFrequency` -- 32000 vs 2000-4000
- `ADBitVolts` -- different A/D scaling factors
- `InputRange` -- 2000, 10000, or 100000 uV
- `DspLowCutFrequency` / `DspHighCutFrequency` -- filter settings
- `ADChannel` -- hardware channel number (maps to combo board slot)

## ATLAS System Architecture

- Up to 16 combo boards, 32 channels each (512 max)
- 24-bit A/D, 40 kHz max per channel, simultaneous sampling
- Each 32-channel bank: 8 local references + 7 global references
- Input-referred noise: 1.3 uV RMS (0.1-9000 Hz)
- Input impedance: ~1 TOhm (adequate for micro electrodes)
- CMRR: 110 dB (direct), 90 dB (programmable reference)

## References

- Nelson, Bhatt et al. "Signal distortion from microelectrodes in clinical EEG
  acquisition systems" (PMC3476479)
- Misra et al. "Methods for implantation of micro-wire bundles" (PMC4019382)
- Staba et al. "Interference and noise in human intracranial microwire recordings"
  (IEEE, 2008)
- Neuralynx ATLAS system documentation: neuralynx.fh-co.com
- Fried Lab setup guide: friedcnl.ucla.edu/docs/neuralynx-system-setup/
